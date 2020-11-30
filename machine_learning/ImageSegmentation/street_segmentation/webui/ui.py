from flask import Flask, render_template
import os
import json

class WebInterface:

    def __init__(self, name, ip_config='0.0.0.0', port=5000):
        # Trainer
        self.trainer = None
        # Ip
        self.ip_config = ip_config
        self.port = port
        # Name
        self.name = name
        # Flask App
        self.app = Flask(name, static_url_path='/webui/static', template_folder='webui/templates/')
        # Call Setup
        self.setup_routes()
        return

    def setup_routes(self):
        # Main route
        self.add_endpoint('/', endpoint_name='main_page', handler=self.render_mainpage, methods=['GET'])
        self.add_endpoint('/webui/static/<file>', endpoint_name='server static files', handler=self.get_static_file,
                          methods=['GET'])
        self.add_endpoint('/train/data/', endpoint_name='train_data', handler=self.get_training_data, methods=['GET'])
        self.add_endpoint('/train', endpoint_name='train_start', handler=self.train_page, methods=['GET'])
        return

    def add_endpoint(self, endpoint=None, endpoint_name=None, handler=None, methods=None):
        self.app.add_url_rule(endpoint, endpoint_name, handler, methods=methods)
        return

    def run(self):
        if self.app:
            self.app.run(host=self.ip_config, port=self.port)
        else:
            print("No app setup which shouldn't be possible")
        return

    def set_trainer(self, trainer):
        self.trainer = trainer
        return

    def get_trainer(self):
        return self.trainer

    def train(self):
        self.trainer.train_network()
        return

    # /train
    def train_page(self):
        self.train()
        return json.dumps({'status':'OK'})

    # /webui/static/<file>
    def get_static_file(self, file):
        from flask import send_from_directory
        static_file = os.path.join(os.getcwd(), 'webui/static/')
        return send_from_directory(static_file, file)

    # /train/data
    def get_training_data(self):
        epoch_loss = self.trainer.epoch_losses
        epoch_test_losses = self.trainer.epoch_test_losses
        test_accs = self.trainer.test_accs
        train_accs = self.trainer.train_accs
        curr_epoch = self.trainer.curr_epoch
        total_epochs = self.trainer.num_epoch
        curr_active = self.trainer.active
        return json.dumps({'status':'OK', 'epoch_loss':epoch_loss, 'epoch_test_losses':epoch_test_losses,
                           'test_accs':test_accs, 'curr_epoch':curr_epoch,'total_epochs':total_epochs,
                           'curr_active':curr_active, 'train_accs': train_accs})

    # Rendering Functions
    # /
    def render_mainpage(self):
        print("Main Page")
        return render_template('main.html', name=self.name, dataset_name=self.trainer.config['dataset']['name'],
                               batch_size=self.trainer.config['dataset']['batch_size'],
                               shuffle=self.trainer.config['dataset']['shuffle'],
                               split=self.trainer.config['dataset']['split'],
                               num_epochs=self.trainer.config['run']['num_epochs'],
                               model=self.trainer.config['models']['name'],
                               num_classes=self.trainer.config['models']['num_classes'],
                               lr=self.trainer.config['optim']['lr'],
                               optim=self.trainer.config['optim']['name'])


