var shoulddisplay = false;

const reset = document.getElementById("scanAnother");
const img = document.getElementById('obb_img');
var video = document.getElementById("webCamera");
var obb_img = document.getElementById("result");


async function resetAndReload() {
    try {
        let currentURL = new URL(window.location.href);
        window.location.href = currentURL.toString();
    } catch (error) {
        console.log('Error resetting the session data: ', error);
    }
}

if (obb_img) {
    reset.addEventListener('click', () => {
        resetAndReload();
    });
};

function formSubmit() {
    document.forms['image_submit'].submit();
    stopStreamedVideo(video);
}

const getFrame = () => {
    const overlay = document.getElementById('overlay');
    overlay.style.display = 'none';
    const preloader = document.getElementById('preloader');
    preloader.style.display = 'block';
    const canvas = document.createElement('canvas');
    var video = document.querySelector("#webCamera");
    var ctx = canvas.getContext('2d');
    var vid_canvas = document.querySelector("#videoCanvas");
    vid_canvas.style.display = "none";
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext('2d').drawImage(video, 0, 0);
    const data = canvas.toDataURL('image/png');
    document.getElementById("frame").value = data;
    formSubmit()
}

function drawImge() {
    var video = document.querySelector("#webCamera");
    var canvas = document.querySelector("#videoCanvas");
    var canvasRect = video.getBoundingClientRect();
    canvas.style.position = "relative";

    var ctx = canvas.getContext('2d');
    canvas.width = window.innerWidth - 150;
    canvas.height = window.innerHeight - 200;
    ctx.beginPath()
}

var constraints = {
    audio: false,
    video: {facingMode: {exact: 'environment'}}
};

var constraints_no_backcam = {
    audio: false,
    video: true
}

function successCallback(stream) {
    window.stream = stream; // make stream available to console
    video.srcObject = stream;
}

function errorCallback(error) {
    console.log('navigator.getUserMedia error: ', error);
    navigator.mediaDevices.getUserMedia(constraints_no_backcam).then(successCallback).catch(errorCallback);
}

console.log(obb_img)
if (! obb_img) {
    // Get the dimensions of the container
    var container = document.getElementById("portfolio");

    // Function to update the circle size
    function updateCircleSize() {
        var containerWidth = container.offsetWidth;
        var containerHeight = container.offsetHeight;
        var circleSize = Math.min(containerWidth, containerHeight) * 0.5;
        var overlay = document.getElementById("overlay");
        overlay.style.width = circleSize + "px";
        overlay.style.height = circleSize + "px";
        overlay.style.marginTop = '36px';
    }

    // Call the function initially
    updateCircleSize();

    // Add event listener for window resize
    window.addEventListener("resize", updateCircleSize);

    // navigator.mediaDevices.getUserMedia({video: {facingMode: {exact: 'environment'}}}).then((stream) => video.srcObject = stream);
    navigator.mediaDevices.getUserMedia(constraints).then(successCallback).catch(errorCallback);

    //video.style.height = '100%';
    video.onplay = function () {
        video.style = "initial"
        video.style.position = 'fixed';
        video.style.top = '0';
        video.style.left = '0';
        video.style.width = '100%';
        video.style.height = '100%';
        setTimeout(drawImge, 100);
        video.style.marginTop = '36px';
    };
}

function stopStreamedVideo(videoElem) {
    const stream = videoElem.srcObject;
    const tracks = stream.getTracks();

    tracks.forEach((track) => {
        track.stop();
    });
    video.style = "none"
    videoElem.srcObject = null;
}
