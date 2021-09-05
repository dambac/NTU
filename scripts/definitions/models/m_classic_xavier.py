import torch

encoder_hidden_size = 2048
classifier_hidden_size = 2048
output_size = 2


class MClassicXavier(torch.nn.Module):

    @staticmethod
    def create(input_size):
        net: MClassicXavier = MClassicXavier(input_size)

        # initialization function, first checks the module type,
        # then applies the desired changes to the weights
        def init_normal(m):
            if type(m) == torch.nn.Linear:
                torch.nn.init.uniform_(m.weight)

        # use the modules apply function to recursively apply the initialization

        net.apply(init_normal)

        return net

    def __init__(self, input_size):
        super(MClassicXavier, self).__init__()

        """
        Siamese Network
        """
        self.siamese_linear1 = torch.nn.Linear(input_size, encoder_hidden_size)
        self.siamese_activation1 = torch.nn.LeakyReLU()

        self.siamese_linear2 = torch.nn.Linear(encoder_hidden_size, encoder_hidden_size)
        self.siamese_activation2 = torch.nn.LeakyReLU()

        self.siamese_linear3 = torch.nn.Linear(encoder_hidden_size, classifier_hidden_size)
        self.siamese_activation3 = torch.nn.LeakyReLU()

        """
        Pooling both
        """
        self.poller = torch.nn.MaxPool1d(2)

        """
        Classifier
        """
        self.class_linear1 = torch.nn.Linear(classifier_hidden_size, classifier_hidden_size)
        self.class_activation1 = torch.nn.LeakyReLU()

        self.class_linear2 = torch.nn.Linear(classifier_hidden_size, classifier_hidden_size)
        self.class_activation2 = torch.nn.LeakyReLU()

        self.class_linear3 = torch.nn.Linear(classifier_hidden_size, output_size)
        self.class_activation3 = torch.nn.LeakyReLU()

    def forward(self, X):
        """
        X.shape = (batch, views, encoding_size)
        """

        X = self.siamese_activation1(self.siamese_linear1(X))
        X = self.siamese_activation2(self.siamese_linear2(X))
        X = self.siamese_activation3(self.siamese_linear3(X))

        X = X.permute(0, 2, 1)
        X = self.poller(X).squeeze(2)

        X = self.class_activation1(self.class_linear1(X))
        X = self.class_activation2(self.class_linear2(X))
        X = self.class_activation3(self.class_linear3(X))

        return X
