import mxnet as mx
from mxnet import gluon
import tarfile
import json
import sklearn

def vocab_from_json(path):
    with open(path) as inp:
        vocab = json.load(inp)
        print('Vocabulary (%d words) loaded from "%s"', len(vocab), path)
        return vocab
    
def model_fn(model_dir):
    """
    Load the gluon model. Called once when hosting service starts.

    :param: model_dir The directory where model files are stored.
    :return: a model (in this case a Gluon network)
    """
    symbol = mx.sym.load('%s/model.json' % model_dir)
    vocab = vocab_from_json('%s/vocab.json' % model_dir)
    outputs = mx.symbol.softmax(data=symbol, name='softmax_label')
    inputs = mx.sym.var('data')
    param_dict = gluon.ParameterDict('model_')
    net = gluon.SymbolBlock(outputs, inputs, param_dict)
    net.load_parameters('%s/model.params' % model_dir, ctx=mx.cpu())
    return net, vocab


def transform_fn(net, data):
    """
    Transform a request using the Gluon model. Called once per request.

    :param net: The Gluon model.
    :param data: The request payload.
    :param input_content_type: The request content type.
    :param output_content_type: The (desired) response content type.
    :return: response payload and content type.
    """
    # we can use content types to vary input/output handling, but
    # here we just assume json for both
    net, vocab = net
    outputs = []
    tokens = [vocab.get(token, 1) for token in data.lower().split()]
    nda = mx.nd.array([tokens])
    output = net(nda)
    response_body = f'Positivity: {output[0][1].asscalar():.0%}'
    return response_body

with tarfile.open('model.tar.gz', "r:gz") as tar:
    tar.extractall('/tmp')
net = model_fn('/tmp')

def lambda_handler(event, context):
    try:
        text = event['text']
        response = transform_fn(net, text)
        return response
    except Exception as e:
        raise Exception('ProcessingError')
