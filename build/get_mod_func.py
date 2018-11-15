import paddle.v2 as api
import importlib
import inspect

for mod_name in (filter(lambda x: x[0]!="_", dir(api.dataset))):
    mod_obj = importlib.import_module("paddle.v2.dataset." + mod_name)
    functions = filter(lambda x: (x[0].startswith("test") or x[0].startswith("train")) \
                           , inspect.getmembers(mod_obj, inspect.isfunction))
 
    for func in functions:
        print mod_name, ":", func[0]
        #print func[1]()
        print inspect.getargspec(func[1])
