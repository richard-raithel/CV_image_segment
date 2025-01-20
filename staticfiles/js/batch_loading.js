(function () {
    const sample_check = document.querySelector("#image-0")
    const reset = document.getElementById("newBatch");

    function getCsrfToken() {
        const tokenElement = document.querySelector('input[name="csrfmiddlewaretoken"]');
        return tokenElement ? tokenElement.value : '';
    }

    function resetSessionData() {
        const csrf_token = getCsrfToken();
        return new Promise((resolve, reject) => {
            fetch(refreshURL, {
                method: 'POST',
                headers: {
                    'X-CSRFToken': csrf_token,
                },
            }).then(response => {
                if (response.ok) {
                    // Handle a successful response
                    console.error('Refreshed session data');
                    resolve();

                } else {
                    // Handle an error response
                    reject()
                }
            }).catch(error => {
                // Handle any errors that occurred during the fetch process
                console.error('Error:', error);
                reject(error);
            });
        });
    }

    async function resetAndReload() {
        try {
            await resetSessionData();
            window.location.reload();
        } catch (error) {
            console.log('Error resetting the session data: ', error);
        }
    }

    reset.addEventListener('click', () => {
        resetAndReload();
    });

    if (!sample_check) {
        const video = document.getElementById('captureVideo');
        const button = document.getElementById('imageCapture');
        const close = document.getElementById('closeCapture');
        const batchcontainer = document.getElementById('batchContainer');
        const upload = document.getElementById("upload");
        const loadingGif = document.getElementById("preloader");
        const samples = [];

        // Get the dimensions of the container
        var container = document.getElementById("captureVideo");

        // Function to update the circle size
        function updateCircleSize() {
            var containerWidth = container.offsetWidth;
            var containerHeight = container.offsetHeight;
            var circleSize = Math.min(containerWidth, containerHeight) * 0.6;
            var overlay = document.getElementById("overlay");
            overlay.style.width = circleSize + "px";
            overlay.style.height = circleSize + "px";
        }

        // Call the function initially
        updateCircleSize();

        // Add event listener for window resize
        window.addEventListener("resize", updateCircleSize);

        reset.style.display = 'none';

        window.onload = function () {
            // Add grid layout styles to batchContainer
            Object.assign(batchContainer.style, {
                display: 'grid',
                gridTemplateColumns: 'repeat(auto-fill, minmax(300px, 1fr))',
            });
        };

        const setFixedStyle = (elem, top, left, width, height) => {
            elem.style.position = 'fixed';
            elem.style.top = top;
            elem.style.left = left;
            elem.style.width = width;
            elem.style.height = height;
        };

        const createSample = (imgData) => {

            // create the outermost row div element
            var row = document.createElement('div');
            row.setAttribute('class', 'row d-flex justify-content-center card-deck');
            row.setAttribute('id', 'card-deck');

            // create the profile card div element
            var sample = document.createElement('div');
            sample.setAttribute('class', 'profile-card-4 card text-center p-0 border-0 shadow');

            // create the img element with its src and alt attributes
            var img = document.createElement('img');
            img.src = imgData;
            // img.setAttribute('src', "{% static 'images/DSC_3317.jpg' %}");
            img.setAttribute('class', 'card-img-top');
            img.setAttribute('alt', '...');
            img.setAttribute('style', 'min-height: 300px');

            // create the div element for the input fields
            var div1 = document.createElement('div');
            div1.setAttribute('class', 'p-3 mt-4');

            // create the profile content div element
            var profileContent = document.createElement('div');
            profileContent.setAttribute('class', 'profile-content');

            // create the row div element for the input fields
            var row2 = document.createElement('div');
            row2.setAttribute('class', 'row');

            // create the first input field
            var col1 = document.createElement('div');
            col1.setAttribute('class', 'col-12 pb-3');
            var formulation = document.createElement('input');
            formulation.setAttribute('type', 'text');
            formulation.setAttribute('placeholder', 'Enter Product Name');
            formulation.setAttribute('id', 'formulation');
            col1.appendChild(formulation);

//            const currentFormulation = formulations.splice(0,1);
//            if (currentFormulation.length > 0) {
//                formulation.value = currentFormulation;
//            }

            // create the second input field
            var col2 = document.createElement('div');
            col2.setAttribute('class', 'col-12 pb-3');
            var rating = document.createElement('input');
            rating.setAttribute('type', 'text');
            rating.setAttribute('placeholder', 'Enter Rating');
            rating.setAttribute('id', 'rating');

            // Add event listener for input event
            rating.addEventListener('input', function(event) {
              var input = event.target;
              var inputValue = input.value;
              // Use regular expression to match integers only
              var regex = /^\d+$/;
              if (!regex.test(inputValue)) {
                // If input value is not an integer, remove non-numeric characters
                inputValue = inputValue.replace(/\D/g, '');
                input.value = inputValue;
              }
            });
            col2.appendChild(rating);

            // append the child elements to their respective parent elements
            row.appendChild(sample);
            sample.appendChild(img);
            sample.appendChild(div1);
            div1.appendChild(profileContent);
            profileContent.appendChild(row2);
            row2.appendChild(col1);
            row2.appendChild(col2);

            return sample;
        };

        const stopStreamedVideo = (videoElem) => {
            const stream = videoElem.srcObject;
            const tracks = stream.getTracks();

            tracks.forEach((track) => {
                track.stop();
            });

            videoElem.srcObject = null;
        };

        const constraints = {
            audio: false,
            video: {facingMode: {exact: 'environment'}},

        };

        const constraints_no_backcam = {
            audio: false,
            video: true,
        };

        const successCallback = (stream) => {
            window.stream = stream;
            video.srcObject = stream;
        };

        const errorCallback = (error) => {
            console.log('navigator.getUserMedia error: ', error);
            navigator.mediaDevices.getUserMedia(constraints_no_backcam).then(successCallback).catch(errorCallback);
        };

        button.addEventListener('click', () => {
            const canvas = document.createElement("canvas");
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;

            const ctx = canvas.getContext('2d');
            ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

            const tmp_canvas = document.createElement('canvas');
            const tmp_ctx = tmp_canvas.getContext('2d');
            tmp_canvas.width = canvas.width * .5;
            tmp_canvas.height = canvas.height * .5;

            tmp_ctx.drawImage(canvas, 0, 0, tmp_canvas.width, tmp_canvas.height);

            const imgData = tmp_canvas.toDataURL();
            const newsample = createSample(imgData);

            batchcontainer.appendChild(newsample);
            samples.push(newsample);
        });

        close.addEventListener('click', () => {
            stopStreamedVideo(video);

            const videoContainer = document.getElementById('videoContainer');
            const batchUpload = document.getElementById('batchUpload');
            videoContainer.style.cssText = 'display: none !important;';
            batchUpload.style.display = 'flex';

            // Position the upload button
            // setFixedStyle(batchUpload, 'calc(0% + 25px)', 'calc(50% - 50px)', '20%', '15%');
        });

        async function sendBatch(chunkIndex, formData, csrf_token) {
            const requestOptions = {
                method: 'POST',
                headers: {
                    'X-CSRFToken': csrf_token,
                },
                body: formData,
            };

            try {
                const response = await fetch(batchSubmitUrl, requestOptions);
                if (response.ok) {
                    const result = await response.json();
                    return { success: true, data: result };
                } else {
                    return { success: false, error: `Error in chunk ${chunkIndex}` };
                }
            } catch (error) {
                return { success: false, error: `Error in chunk ${chunkIndex}` };
            }
        }

        async function submitForm() {
            const csrf_token = getCsrfToken();

            const chunkSize = 10;
            const totalChunks = Math.ceil(samples.length / chunkSize);
            const resultContainer = document.querySelector('.result-container');

            const formData = new FormData();
            for (let i = 0; i < samples.length; i++) {

                const formulationInput = samples[i].querySelector('#formulation');
                const ratingInput = samples[i].querySelector('#rating');

                const formulationValue = formulationInput ? formulationInput.value || "Not provided" : "Not provided";
                const ratingValue = ratingInput ? ratingInput.value || "Not provided" : "Not provided";


                const imgSrc = samples[i].querySelector('img').src;
                formData.append(`samples[${i}][img_data]`, imgSrc);
                formData.append(`samples[${i}][formulation]`, formulationValue);
                formData.append(`samples[${i}][rating]`, ratingValue);
            }



//
//            // Create a new promise for each request
//            const xhr = new XMLHttpRequest();
//            xhr.open("POST", batchSubmitUrl, true)
//            xhr.setRequestHeader('X-CSRFToken', csrf_token)
//
//            xhr.onload = function () {
//                if (xhr.status === 200) {
//                    console.log("Success");
//                    window.location.reload()
//                } else {
//                    console.error("Error");
//                    window.location.reload()
//                }
//            };
//
//            xhr.onerror = function () {
//                console.error("Error");
//            };
//
//            await xhr.send(formData);

            // Reload the page after all uploads are complete
        }

        upload.addEventListener('click', () => {
            upload.style.display = 'none';
            batchcontainer.style.display = 'none';
            loadingGif.style.display = 'block';
            Object.assign(loadingGif.style, {
                position: 'fixed',
                top: '50%',
                left: '50%',
                width: '50%',
                height: '50%',
                transform: 'translate(-50%, -50%)',
                display: 'flex',
                justifyContent: 'center',
                padding: '0 20px'
            });
            submitForm();
        });

        video.addEventListener('play', () => {
            const videoContainer = document.getElementById('videoContainer');
            const buttonWrapper = document.getElementById('buttonWrapper');
            const uploadButton = document.getElementById('batchUpload');

            uploadButton.style.display = 'none';
            video.style = "initial";
            // Object.assign(videoContainer.style, { position: 'fixed', top: '50%', left: '50%', width: '50%', height: '50%', transform: 'translate(-50%, -50%)', overflow: 'hidden' });
            Object.assign(buttonWrapper.style, {
                position: 'fixed',
                bottom: '10px',
                left: '50%',
                width: '50%',
                height: '10%',
                display: 'flex',
                justifyContent: 'center',
                padding: '0 20px'
            });
        });

        navigator.mediaDevices.getUserMedia(constraints).then(successCallback).catch(errorCallback);
    } else {
        reset.style.display = 'block';
    }

})();
