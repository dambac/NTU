import torch

encoder_hidden_size = 2048
classifier_hidden_size = 2048
output_size = 2


class MClassicDrop1(torch.nn.Module):

    @staticmethod
    def create(input_size):
        return MClassicDrop1(input_size)

    def __init__(self, input_size):
        super(MClassicDrop1, self).__init__()

        """
        Siamese Network
        """
        self.siamese_linear1 = torch.nn.Linear(input_size, encoder_hidden_size)
        self.siamese_drop1 = torch.nn.Dropout(p=0.8)
        self.siamese_activation1 = torch.nn.LeakyReLU()

        self.siamese_linear2 = torch.nn.Linear(encoder_hidden_size, encoder_hidden_size)
        self.siamese_drop2 = torch.nn.Dropout(p=0.5)
        self.siamese_activation2 = torch.nn.LeakyReLU()

        self.siamese_linear3 = torch.nn.Linear(encoder_hidden_size, classifier_hidden_size)
        self.siamese_drop3 = torch.nn.Dropout(p=0.5)
        self.siamese_activation3 = torch.nn.LeakyReLU()

        """
        Pooling both
        """
        self.poller = torch.nn.MaxPool1d(2)

        """
        Classifier
        """
        self.class_linear1 = torch.nn.Linear(classifier_hidden_size, classifier_hidden_size)
        self.class_drop1 = torch.nn.Dropout(p=0.5)
        self.class_activation1 = torch.nn.LeakyReLU()

        self.class_linear2 = torch.nn.Linear(classifier_hidden_size, classifier_hidden_size)
        self.class_drop2 = torch.nn.Dropout(p=0.5)
        self.class_activation2 = torch.nn.LeakyReLU()

        self.class_linear3 = torch.nn.Linear(classifier_hidden_size, output_size)
        self.class_activation3 = torch.nn.LeakyReLU()

    def forward(self, X):
        """
        X.shape = (batch, views, encoding_size)
        """

        X = self.siamese_drop1(self.siamese_activation1(self.siamese_linear1(X)))
        X = self.siamese_drop2(self.siamese_activation2(self.siamese_linear2(X)))
        X = self.siamese_drop3(self.siamese_activation3(self.siamese_linear3(X)))

        X = X.permute(0, 2, 1)
        X = self.poller(X).squeeze(2)

        X = self.class_drop1(self.class_activation1(self.class_linear1(X)))
        X = self.class_drop2(self.class_activation2(self.class_linear2(X)))
        X = self.class_activation3(self.class_linear3(X))

        return X
