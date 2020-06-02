import paddle_serving_client.io as serving_io
serving_io.inference_model_to_serving("./ckpt_20200601082108/best_model", serving_server="serving_server", serving_client="serving_client")
