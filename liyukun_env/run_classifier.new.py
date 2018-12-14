import os
import time
import numpy as np

import paddle
import paddle.fluid as fluid
from numpy import linalg as LA
#from load_params_from_tf import parse


from reader_tx import XnliDataReader, TFXnliDataReader
from model import BertModel
from predict import predict_wrapper
from utils import parse_args
from utils import print_arguments
from utils import init_pretraining_model
from utils import append_nccl2_prepare


def create_model(pyreader_name, is_training=True):
    pyreader = fluid.layers.py_reader(
        capacity=50,
        shapes=[[-1, args.max_seq_len, 1], [-1, args.max_seq_len, 1],
                [-1, args.max_seq_len, 1],
                [-1, args.num_head, args.max_seq_len],
                [-1, 1], [-1, 1]],
        dtypes=['int64', 'int64', 'int64', 'float', 'int64', 'int64'],
        lod_levels=[0, 0, 0, 0, 0, 0],
        name=pyreader_name,
        use_double_buffer=True)

    (src_ids, pos_ids, sent_ids, self_attn_mask, labels,
     next_sent_index) = fluid.layers.read_file(pyreader)

    bert = BertModel(
        src_ids=src_ids,
        position_ids=pos_ids,
        sentence_ids=sent_ids,
        self_attn_mask=self_attn_mask,
        emb_size=args.d_model,
        n_layer=args.num_layers,
        n_head=args.num_head,
        voc_size=args.vocab_size,
        max_position_seq_len=512,
        weight_sharing=args.weight_sharing,
        pad_sent_id=2)

    cls_feats = bert.get_pooled_output(next_sent_index)
    cls_feats = fluid.layers.dropout(x=cls_feats, dropout_prob=0.1)
    logits = fluid.layers.fc(
        input=cls_feats,
        size=3,
        param_attr=fluid.ParamAttr(
            name="cls_out_w",
            initializer=fluid.initializer.TruncatedNormal(scale=0.02)),
        bias_attr=fluid.ParamAttr(
            name="cls_out_b", initializer=fluid.initializer.Constant(0.)))
    ce_loss, probs = fluid.layers.softmax_with_cross_entropy(
        logits=logits, label=labels, return_softmax=True)
    loss = fluid.layers.reduce_mean(input=ce_loss)

    num_seqs = fluid.layers.create_tensor(dtype='int64')
    accuracy = fluid.layers.accuracy(input=probs, label=labels, total=num_seqs)

    loss.persistable = True
    probs.persistable = True
    accuracy.persistable = True
    num_seqs.persistable = True
    return pyreader, loss, probs, accuracy, num_seqs


def evaluate(test_exe, test_program, test_pyreader, fetch_list):
    test_pyreader.start()
    total_cost, total_acc, total_num_seqs = [], [], []
    time_begin = time.time()
    while True:
        try:
            np_loss, np_acc, np_num_seqs = test_exe.run(fetch_list=fetch_list)
            total_cost.extend(np_loss * np_num_seqs)
            total_acc.extend(np_acc * np_num_seqs)
            total_num_seqs.extend(np_num_seqs)
        except fluid.core.EOFException:
            test_pyreader.reset()
            break
    time_end = time.time()
    print("[evalutaion] ave loss: %f, ave_acc: %f, elapsed time: %f s" %
          (np.sum(total_cost) / np.sum(total_num_seqs),
           np.sum(total_acc) / np.sum(total_num_seqs), time_end - time_begin))


def train(args):
    train_program = fluid.Program()
    train_startup = fluid.Program()
    with fluid.program_guard(train_program, train_startup):
        with fluid.unique_name.guard():
            train_pyreader, loss, probs, accuracy, num_seqs = create_model(
                pyreader_name='train_reader')
            
            if args.warmup_steps > 0:
                lr = fluid.layers.learning_rate_scheduler\
                     .noam_decay(1/(args.warmup_steps *(args.learning_rate ** 2)),
                                 args.warmup_steps)
                optimizer = fluid.optimizer.Adam(learning_rate=lr)
            else:
                optimizer = fluid.optimizer.Adam(
                    learning_rate=args.learning_rate)

            fluid.clip.set_gradient_clip(clip=fluid.clip.GradientClipByGlobalNorm(clip_norm=1.0))

            param_list = [param * 1.0 for param in train_program.global_block().all_parameters()]
            
            optimizer.minimize(loss)
              
            if args.weight_decay > 0:
                def exclude_from_weight_decay(name):
                    excluded = name.find("layer_norm")
                    bias_suffix = ["_bias", "_b", ".b_0"]
                    for suffix in bias_suffix:
                        excluded = excluded or name.endswith(suffix)
                    return excluded
                for index, param in enumerate(train_program.global_block().all_parameters()):
                    if not exclude_from_weight_decay(param.name):
                        updated_param = param - param_list[index] * args.weight_decay * (lr if args.warmup_steps > 0 else args.learning_rate)
                        fluid.layers.assign(output=param, input=updated_param)
        
            fluid.memory_optimize(train_program)

    #print(train_program)
    test_prog = fluid.Program()
    test_startup = fluid.Program()
    with fluid.program_guard(test_prog, test_startup):
        with fluid.unique_name.guard():
            test_pyreader, loss, probs, accuracy, num_seqs = create_model(
                pyreader_name='test_reader')

    test_prog = test_prog.clone(for_test=True)

    if args.use_cuda:
        place = fluid.CUDAPlace(0)
        dev_count = fluid.core.get_cuda_device_count()
    else:
        place = fluid.CPUPlace()
        dev_count = int(os.environ.get('CPU_NUM', multiprocessing.cpu_count()))

    print("device count %d" % dev_count)
    print("theoretical memory usage: ")
    #print(fluid.contrib.memory_usage(
    #    program=train_program, batch_size=args.batch_size // args.max_seq_len))

    nccl2_num_trainers = 1
    nccl2_trainer_id = 0
    print("args.is_distributed:", args.is_distributed)
    if args.is_distributed:
        trainer_id = int(os.getenv("PADDLE_TRAINER_ID", "0"))
        port = os.getenv("PADDLE_PORT")
        worker_ips = os.getenv("PADDLE_TRAINERS")
        worker_endpoints = []
        for ip in worker_ips.split(","):
            worker_endpoints.append(':'.join([ip, port]))
        trainers_num = len(worker_endpoints)
        current_endpoint = os.getenv("POD_IP") + ":" + port
        if trainer_id == 0:
            print("train_id == 0, sleep 60s")
            time.sleep(60)
        print("trainers_num:{}".format(trainers_num))
        print("worker_endpoints:{}".format(worker_endpoints))
        print("current_endpoint:{}".format(current_endpoint))

        print("prepare nccl2")
        append_nccl2_prepare(train_startup, trainer_id, worker_endpoints,
                             current_endpoint)
        nccl2_num_trainers = trainers_num
        nccl2_trainer_id = trainer_id

    place = fluid.CUDAPlace(0) if args.use_cuda == True else fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(train_startup)
    exe.run(test_startup)

    """
    tf_fluid_param_map, tf_param_name_value_map = parse("./chinese_L-12_H-768_A-12/bert_model.ckpt")
    for tf_param in tf_fluid_param_map:
      fluid_param_name = tf_fluid_param_map[tf_param]
      fluid_param_value = tf_param_name_value_map[tf_param]
      fluid.global_scope().find_var(fluid_param_name).get_tensor().set(fluid_param_value, place)
      print("init param %s" % (fluid_param_name))
    save_path = os.path.join("params_from_tf")
    fluid.io.save_params(exe, save_path, train_program)
    """
    if args.init_model and args.init_model != "":
        init_pretraining_model(
            exe,
            pretraining_model_path=args.init_model,
            main_program=train_program)
   
    for param in train_program.global_block().all_parameters():
        print(param.name, param.shape, LA.norm(np.array(fluid.global_scope().find_var(param.name).get_tensor()))) 
    """
    data_reader = XnliDataReader(
        data_dir=args.data_dir,
        vocab_path=args.vocab_path,
        batch_size=args.batch_size,
        voc_size=args.vocab_size,
        pad_word_id=args.vocab_size,
        epoch=args.epoch,
        mask_id=-1,  # no mask
        pad_sent_id=args.pad_sent_id,
        max_seq_len=args.max_seq_len,
        num_head=args.num_head)
    """
    tf_xnli_reader =  TFXnliDataReader(
                 data_dir=args.data_dir,
                 vocab_path=args.vocab_path,
                 batch_size=args.batch_size,
                 max_seq_len=args.max_seq_len,
                 shuffle_files=True,
                 epoch=100,
                 pad_word_id=0,
                 pad_sent_id=0,
                 pad_pos_id=0,
                 mask_id=-1)

    exec_strategy = fluid.ExecutionStrategy()
    if args.use_fast_executor:
        exec_strategy.use_experimental_executor = True

    build_strategy = fluid.BuildStrategy()
    build_strategy.remove_unnecessary_lock = True

    train_exe = fluid.ParallelExecutor(
        use_cuda=args.use_cuda,
        loss_name=loss.name,
        build_strategy=build_strategy,
        exec_strategy=exec_strategy,
        main_program=train_program,
        num_trainers=nccl2_num_trainers,
        trainer_id=nccl2_trainer_id)

    test_exe = fluid.ParallelExecutor(
        use_cuda=args.use_cuda,
        main_program=test_prog,
        share_vars_from=train_exe)

    train_pyreader.decorate_tensor_provider(
        tf_xnli_reader.data_generator(
            lang='zh', phase='train'))
    test_pyreader.decorate_tensor_provider(
        tf_xnli_reader.data_generator(
            lang='zh', phase='dev'))

    train_pyreader.start()
    steps = 0
    total_cost, total_acc, total_num_seqs = [], [], []
    time_begin = time.time()
    while True:
        try:
            if args.warmup_steps <= 0:
                np_loss, np_acc, np_num_seqs = train_exe.run(
                    fetch_list=[loss.name, accuracy.name, num_seqs.name])
            else:
                np_loss, np_acc, np_lr, np_num_seqs = train_exe.run(
                    fetch_list=[
                        loss.name, accuracy.name, lr.name, num_seqs.name
                    ])
            total_cost.extend(np_loss * np_num_seqs)
            total_acc.extend(np_acc * np_num_seqs)
            total_num_seqs.extend(np_num_seqs)
            steps += 1
            #print("learning rate", np.array(fluid.global_scope().find_var("learning_rate_0").get_tensor())) 
            if steps % args.skip_steps == 0:
                #print("feed_queue size", train_pyreader.queue.size())
                time_end = time.time()
                used_time = time_end - time_begin
                epoch, current_file_index, total_file, current_file = tf_xnli_reader.get_progress(
                )
                if args.warmup_steps > 0:
                    print("current learning rate: %f" % np_lr[0])
                print(
                    "epoch: %d, progress: %d/%d, step: %d, loss: %f, cls_acc: %f, speed: %f steps/s, file: %s"
                    % (epoch, current_file_index, total_file, steps,
                       np.sum(total_cost) / np.sum(total_num_seqs),
                       np.sum(total_acc) / np.sum(total_num_seqs),
                       args.skip_steps / used_time, current_file))
                total_cost, total_acc, total_num_seqs = [], [], []
                time_begin = time.time()

            if steps % args.save_steps == 0:
                save_path = os.path.join(args.checkpoints,
                                         "step_" + str(steps))
                fluid.io.save_persistables(exe, save_path, train_program)

            if steps % args.validation_steps == 0:
                evaluate(test_exe, test_prog, test_pyreader,
                         [loss.name, accuracy.name, num_seqs.name])

        except fluid.core.EOFException:
            save_path = os.path.join(args.checkpoints, "step_" + str(steps))
            fluid.io.save_persistables(exe, save_path, train_program)
            train_pyreader.reset()
            break
    evaluate(test_exe, test_prog, test_pyreader,
             [loss.name, accuracy.name, num_seqs.name])


if __name__ == '__main__':
    args = parse_args()
    print_arguments(args)
    if args.for_test:
        test(args)
    else:
        train(args)
