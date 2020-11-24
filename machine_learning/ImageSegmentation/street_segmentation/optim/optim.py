from torch import optim

supported_optims = ['Adam', 'SGD', 'Adagrad','RMSProp']

def load_optim(optimizer, model, custom_optim=None):
    run_optim = ""
    optim_name = optimizer['name']
    lr = optimizer['lr']
    if optim_name in supported_optims:
        if optim_name == "Adam":
            run_optim = optim.Adam(model.parameters(), lr=lr)
        elif optim_name == "SGD":
            run_optim = optim.SGD(model.parameters(), lr=lr)
        elif optim_name == "Adagrad":
            run_optim = optim.Adagrad(model.parameters(), lr=lr)
        elif optim_name == "RMSProp":
            run_optim = optim.RMSProp(model.parameters(), lr=lr)
        else:
            print("Unknown optim defaulting to...Adam")
            run_optim = optim.Adam(model.parameters(), lr=lr)
    else:
        run_optim = custom_optim
    return run_optim