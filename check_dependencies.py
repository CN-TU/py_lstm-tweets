import pip

def install(package):
    pip.main(['install', package])

def install_all_packages(modules_to_try):
    for module in modules_to_try:
        try:
           __import__(module)        
        except ImportError as e:
            install(e.name) 

a = ['numpy', 'pandas', 'fileinput', 'sys', 're', 'csv', 'datasketch', 'collections', 'pickle', 'keras.preprocessing', 'keras.models', 'keras.layers']
install_all_packages(a)

 
