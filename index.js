const cv2 = require('/nodecv/node_modules/opencv4nodejs');
const tf = require('@tensorflow/tfjs-node');
const blazeface = require('@tensorflow-models/blazeface');

const FPS = 30;
const Vcap = new cv2.VideoCapture(0);
Vcap.set(cv2.CAP_PROP_FRAME_WIDTH, 300);
Vcap.set(cv2.CAP_PROP_FRAME_HEIGHT, 300);

function loadModel() {
    return new Promise(async(resolve, reject) => {
        try {
            const model = await blazeface.load();
            resolve(model);
        } catch (error) {
            console.log(error.mesage);  
        }
    });
};

loadModel().then((model)=>{
    const intvl = setInterval(() => {
        try {
            const frame = Vcap.read();
    
            if (frame.empty) {
                Vcap.reset();
                frame = Vcap.read();
            }
    
            try {
                function detectFaces(){
                    return new Promise(async(resolve, reject) => {
                        const cvt_img = frame.cvtColor(cv2.COLOR_BGR2RGB);
                        const width = cvt_img.rows;
                        const height = cvt_img.cols;
                        const image = tf.tensor3d(new Uint8Array(cvt_img.getData()), [width, height, 3], 'int32');

                        const prediction = await model.estimateFaces(image, false);
                        if (prediction.length > 0) {
                            resolve(prediction);
                        }
                    });
                };
            } catch (error) {
                console.log(error.mesage);
            }
            
            detectFaces().then((prediction)=>{
                for (let i = 0; i < prediction.length; i++) {
                    frame.drawRectangle(new cv2.Rect(
                        prediction[i].topLeft[0],
                        prediction[i].topLeft[1],
                        prediction[i].bottomRight[0] - prediction[i].topLeft[0],
                        prediction[i].bottomRight[1] - prediction[i].topLeft[1]
                    ), new cv2.Vec3(255, 0, 255), 2);
                };
                cv2.imshow("Window", frame);
            });
            
        } catch (error) {
            console.log(error.mesage);
        };
        
        const key = cv2.waitKey(1);
        if(key == 27){
            clearTimeout(intvl);
            cv2.destroyAllWindows();
        };
    }, 1000 / FPS);
});