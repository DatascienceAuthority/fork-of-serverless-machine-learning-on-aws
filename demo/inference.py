from mxnet.image import imdecode
from gluoncv import model_zoo, data, utils
import requests
from io import BytesIO
import json
import base64
import sklearn

#model = pickle.load(open('model.pkl', 'rb'))


net = model_zoo.get_model('yolo3_mobilenet1.0_coco', pretrained=True, root='/tmp/')

def lambda_handler(event, context):
    try:
        url = event['img_url']
        response = requests.get(url)
        img = imdecode(response.content)
        x, img = data.transforms.presets.yolo.transform_test([img], short=320)
        class_IDs, scores, bounding_boxs = net(x)
        output = utils.viz.plot_bbox(img, bounding_boxs[0], scores[0],
                            class_IDs[0], class_names=net.classes)
        output.axis('off')
        f = BytesIO()
        output.figure.savefig(f, format='jpeg', bbox_inches='tight')
        return base64.b64encode(f.getvalue())
    except Exception as e:
        raise Exception('ProcessingError')
