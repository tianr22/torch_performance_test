class ModelInfo(object):
    def __init__(self, size=7):
        self.model_size = size
        # llama 2
        if size == 7:
            self.num_layers = 32
            self.heads = 32
            self.hidden_dim = 4096
            self.ffn = 11088
            self.heads_k = self.heads
        elif size == 13:
            self.num_layers = 40
            self.heads = 40
            self.hidden_dim = 5120
            self.ffn = 13696
            self.heads_k = self.heads
        elif size == 70:
            self.num_layers = 80
            self.heads = 64
            self.hidden_dim = 8192
            self.ffn = 28672
            self.heads_k = 8
        # llama 3
        elif size == 8:  #
            self.num_layers = 32
            self.heads = 32
            self.hidden_dim = 4096
            self.ffn = 14336
            self.heads_k = 8
