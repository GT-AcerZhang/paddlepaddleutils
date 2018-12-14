import os
import time
import numpy as np

import paddle
import paddle.fluid as fluid

from reader_tx import DataReader
from model import BertModel
from predict import predict_wrapper
from utils import parse_args
from utils import print_arguments
from utils import init_model
from utils import append_nccl2_prepare


def create_model(pyreader_name):
    pyreader = fluid.layers.py_reader(
        capacity=70,
        shapes=[[-1, args.max_seq_len, 1], [-1, args.max_seq_len, 1],
                [-1, args.max_seq_len, 1],
                [-1, args.max_seq_len, args.max_seq_len],
                [-1, 1], [-1, 1], [-1, 1], [-1, 1]],
        dtypes=[
            'int64', 'int64', 'int64', 'float', 'int64', 'int64', 'int64',
            'int64'
        ],
        lod_levels=[0, 0, 0, 0, 0, 0, 0, 0],
        name=pyreader_name,
        use_double_buffer=True)

    (src_ids, pos_ids, sent_ids, self_attn_mask, mask_label, mask_pos, labels,
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
        max_position_seq_len=args.max_seq_len,
        pad_sent_id=args.pad_sent_id,
        weight_sharing=args.weight_sharing)

    next_sent_acc, mask_lm_loss, total_loss = bert.get_pretraining_output(
        mask_label, mask_pos, labels, next_sent_index)
    
    next_sent_acc.persistable = True 
    mask_lm_loss.persistable = True
    total_loss.persistable = True 

    return pyreader, next_sent_acc, mask_lm_loss, total_loss


def test(args):
    test_prog = fluid.Program()
    test_startup = fluid.Program()
    with fluid.program_guard(test_prog, test_startup):
        with fluid.unique_name.guard():
            pyreader, next_sent_acc, mask_lm_loss, total_loss = create_model(
                pyreader_name='test_reader')

    test_prog = test_prog.clone(for_test=True)

    place = fluid.CUDAPlace(0) if args.use_cuda == True else fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(test_startup)

    predict = predict_wrapper(
        args,
        test_prog=test_prog,
        pyreader=pyreader,
        fetch_list=[next_sent_acc.name, mask_lm_loss.name, total_loss.name])

    print("test begin")
    loss, lm_loss, acc, steps, speed = predict()
    print(
        "[test_set] loss: %f, global ppl: %f, next_sent_acc: %f, speed: %f steps/s"
        % (np.mean(np.array(loss) / steps),
           np.exp(np.mean(np.array(lm_loss) / steps)),
           np.mean(np.array(acc) / steps), speed))


def train(args):
    print("train begin")
    train_program = fluid.Program()
    train_startup = fluid.Program()
    with fluid.program_guard(train_program, train_startup):
        with fluid.unique_name.guard():
            train_pyreader, next_sent_acc, mask_lm_loss, total_loss = create_model(
                pyreader_name='train_reader')

            lr_decay = fluid.layers.learning_rate_scheduler.noam_decay(
                    args.d_model, args.warmup_steps)
            
            print("before adam")
            with fluid.default_main_program()._lr_schedule_guard():
                learning_rate = args.learning_rate * lr_decay

            optimizer = fluid.optimizer.Adam(learning_rate=learning_rate)
            
            fluid.clip.set_gradient_clip(
                    clip=fluid.clip.GradientClipByGlobalNorm(clip_norm=1.0))
           
            #before minimize
            param_list = None
            if args.weight_decay > 0:
                param_list = [param * 1.0 for param in train_program.global_block().all_parameters()]

            optimizer.minimize(total_loss)
              
            if args.weight_decay > 0:
                def exclude_from_weight_decay(name):
                    if "layer_norm" in name or "word_embedding" in name:
                        return True
                    
                    bias_suffix = ["_bias", "_b", ".b_0"]
                    for suffix in bias_suffix:
                        if name.endswith(suffix):
                            return True
                    return False
                
                for index, param in enumerate(train_program.global_block().all_parameters()):
                    if not exclude_from_weight_decay(param.name):
                        updated_param = param - param_list[index] * args.weight_decay * (learning_rate if args.warmup_steps > 0 else args.learning_rate)
                        fluid.layers.assign(output=param, input=updated_param)
                    else:
                        print("exclude %s from weight decay." % param.name)

            fluid.memory_optimize(train_program)

    test_prog = fluid.Program()
    test_startup = fluid.Program()
    with fluid.program_guard(test_prog, test_startup):
        with fluid.unique_name.guard():
            test_pyreader, next_sent_acc, mask_lm_loss, total_loss = create_model(
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
    print(fluid.contrib.memory_usage(
        program=train_program, batch_size=args.batch_size // args.max_seq_len))

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
        worker_endpoints_env=",".join(worker_endpoints)
        trainers_num = len(worker_endpoints)
        current_endpoint = os.getenv("POD_IP") + ":" + port
        if trainer_id == 0:
            print("train_id == 0, sleep 60s")
            time.sleep(60)
        print("trainers_num:{}".format(trainers_num))
        print("worker_endpoints:{}".format(worker_endpoints))
        print("current_endpoint:{}".format(current_endpoint))

        print("prepare nccl2")
        #append_nccl2_prepare(train_startup, trainer_id, worker_endpoints,
        #                     current_endpoint)
        config = fluid.DistributeTranspilerConfig()
        config.mode = "nccl2"
        t = fluid.DistributeTranspiler(config=config)
        t.transpile(
                trainer_id,
                trainers=worker_endpoints_env,
                current_endpoint=current_endpoint,
                program=train_program,
                startup_program=train_startup)
        nccl2_num_trainers = trainers_num
        nccl2_trainer_id = trainer_id

    place = fluid.CUDAPlace(0) if args.use_cuda == True else fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(train_startup)
    exe.run(test_startup)

    if args.init_model and args.init_model != "":
        # init_model(exe, args.init_model, train_startup)
        init_model(args.init_model, train_program)

    data_reader = DataReader(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        voc_size=args.vocab_size,
        pad_word_id=args.vocab_size,
        epoch=args.epoch,
        pad_sent_id=args.pad_sent_id,
        max_seq_len=args.max_seq_len,
        num_head=args.num_head,
        generate_neg_sample=args.generate_neg_sample)

    exec_strategy = fluid.ExecutionStrategy()
    if args.use_fast_executor:
        exec_strategy.use_experimental_executor = True

    build_strategy = fluid.BuildStrategy()
    build_strategy.remove_unnecessary_lock = False

    train_exe = fluid.ParallelExecutor(
        use_cuda=args.use_cuda,
        loss_name=total_loss.name,
        build_strategy=build_strategy,
        exec_strategy=exec_strategy,
        main_program=train_program,
        num_trainers=nccl2_num_trainers,
        trainer_id=nccl2_trainer_id)

    if args.validation_set_dir and args.validation_set_dir != "":
        predict = predict_wrapper(
            args,
            test_prog=test_prog,
            train_exe=train_exe,
            pyreader=test_pyreader,
            fetch_list=[
                next_sent_acc.name, mask_lm_loss.name, total_loss.name
            ])

    train_pyreader.decorate_tensor_provider(data_reader.data_generator())
    train_pyreader.start()
    steps = 0
    cost = []
    lm_cost = []
    acc = []
    time_begin = time.time()
    while True:
        try:
            each_next_acc, each_mask_lm_cost, each_total_cost = train_exe.run(
                fetch_list=[
                    next_sent_acc.name, mask_lm_loss.name, total_loss.name
                ])
            acc.extend(each_next_acc)
            lm_cost.extend(each_mask_lm_cost)
            cost.extend(each_total_cost)
            steps += nccl2_num_trainers
            
            if nccl2_trainer_id != 0:
                continue
           
            skip_steps = args.skip_steps * nccl2_num_trainers
            if steps % skip_steps == 0:
                print("feed_queue size", train_pyreader.queue.size())
                time_end = time.time()
                used_time = time_end - time_begin
                epoch, current_file_index, total_file, current_file = data_reader.get_progress(
                )
                print(
                    "epoch: %d, progress: %d/%d, step: %d, loss: %f, "
                    "ppl: %f, next_sent_acc: %f, speed: %f steps/s, file: %s"
                    % (epoch, current_file_index, total_file, steps,
                       np.mean(np.array(cost)),
                       np.mean(np.exp(np.array(lm_cost))),
                       np.mean(np.array(acc)), skip_steps / used_time,
                       current_file))
                cost = []
                lm_cost = []
                acc = []
                time_begin = time.time()

            if steps % args.save_steps == 0:
                save_path = os.path.join(args.checkpoints,
                                         "step_" + str(steps))
                fluid.io.save_persistables(exe, save_path, train_program)

            if args.validation_set_dir and steps % args.validation_steps == 0:
                vali_cost, vali_lm_cost, vali_acc, vali_steps, vali_speed = predict(
                )
                print("[validation_set] epoch: %d, step: %d, "
                      "loss: %f, global ppl: %f, batch-averged ppl: %f, "
                      "next_sent_acc: %f, speed: %f steps/s" %
                      (epoch, vali_steps,
                       np.mean(np.array(vali_cost) / vali_steps),
                       np.exp(np.mean(np.array(vali_lm_cost) / vali_steps)),
                       np.mean(np.exp(np.array(vali_lm_cost) / vali_steps)),
                       np.mean(np.array(vali_acc) / vali_steps), vali_speed))

        except fluid.core.EOFException:
            train_pyreader.reset()
            break


if __name__ == '__main__':
    args = parse_args()
    print_arguments(args)
    if args.for_test:
        test(args)
    else:
        train(args)
