import boto3
import cv2
import time

capture = cv2.VideoCapture(0)
if __name__ == '__main__':
    filepath='/Users/lvhaoran/AWScode/Shelf-product-identification/customer/images/logitech.jpg'
    while (True):
        # 获取一帧
        ret, frame = capture.read()
        # 将这帧转换为灰度图
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow('frame', frame)
        # 如果按键q则拍照并跳出本次循环
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    # cv2.imwrite(filepath, frame)
    body = cv2.imencode(".jpg", frame)[1].tobytes()
    # body = b""
    # with open(filepath, "rb") as fp:
    #     body = fp.read()
    # print(body==bs)
    runtime = boto3.client("sagemaker-runtime",region_name="us-east-2")
    tic = time.time()

    # imread then resize then save
    # a = cv2.imread("./endpoint/1.jpeg")
    # a_resize = cv2.resize(a, (100,100))
    # cv2.imwrite("./endpoint/test.jpeg",a_resize)

    response = runtime.invoke_endpoint(
        EndpointName='xinhe',
        Body=body,
        ContentType='image/jpeg',
    )
    body = response["Body"].read()

    toc = time.time()

    print(body.decode())
    print(f"elapsed: {(toc - tic) * 1000.0} ms")
    # cmd1='ls'
    # cmd = 'env PYTHONPATH="/Users/lvhaoran/AWScode/Shelf-product-identification" /Users/lvhaoran/opt/anaconda3/envs/tf1/bin/python -u object_detector_retinanet/keras_retinanet/bin/predict.py single_image --image_path "/Users/lvhaoran/AWScode/Shelf-product-identification/customer/images/mmexport1622530345298.jpg" "/Users/lvhaoran/AWScode/Shelf-product-identification/customer/snapshot/final/iou_resnet50_csv_01.h5" --hard_score_rate=0.5 | tee predict.log'
    # res = os.popen(cmd)
    # output_str = res.read()
    # print(output_str)
    # res=cv2.imread('/Users/lvhaoran/AWScode/Shelf-product-identification/customer/res_images_iou/0.png')
    # cv2.imshow('res', res)
    # cv2.waitKey(10000)