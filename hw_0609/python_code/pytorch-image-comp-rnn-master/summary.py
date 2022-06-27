import argparse
from torchsummary import summary
#from torchinfo import summary

parser = argparse.ArgumentParser()
parser.add_argument('--network', type=str, default='conv', help='type of network')
args = parser.parse_args()

## load networks on GPU
if args.network == 'conv':
    import conv_network

    encoder = conv_network.EncoderCell().cuda()
    binarizer = conv_network.Binarizer().cuda()
    decoder = conv_network.DecoderCell().cuda()

    print("encoder")
    print(summary(encoder, (3, 32, 32)))
    print("binarizer")
    print(summary(binarizer, (512, 2, 2)))
    print("decoder")
    print(summary(decoder, (32, 2, 2)))

elif args.network =='lstm':
    import network_none as network

    encoder = network.EncoderCell().cuda()
    binarizer = network.Binarizer().cuda()
    decoder = network.DecoderCell().cuda()

    print("encoder")
    print(summary(encoder, [(3, 32, 32),
        (256, 8, 8), (512, 4, 4), (512, 2, 2)]))
    print("binarizer")
    print(summary(binarizer, (512, 2, 2)))
    print("decoder")
    print(summary(decoder, [(32, 2, 2),
        (512, 2, 2), (512, 4, 4), (256, 8, 8), (128, 16, 16)]))
