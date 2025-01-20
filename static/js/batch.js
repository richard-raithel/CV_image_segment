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
        const show = document.getElementById('showVideo');
        const batchcontainer = document.getElementById('batchContainer');
        const upload = document.getElementById("upload");
        const loadingGif = document.getElementById("preloader");
        let samples = [];

        // Get the dimensions of the container
        var container = document.getElementById("videoContainer");

        // Function to update the circle size
        function updateCircleSize() {
            var containerWidth = container.offsetWidth;
            var containerHeight = container.offsetHeight;
            var circleSize = Math.min(containerWidth, containerHeight) * 0.75;
            var overlay = document.getElementById("overlay");
            overlay.style.width = circleSize + "px";
            overlay.style.height = circleSize + "px";
        }

        // Add event listener for window resize
        window.addEventListener("resize", updateCircleSize);

        reset.style.display = 'none';

        window.onload = function () {
            // Call the function initially
            setTimeout(updateCircleSize, 500);

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

        const createSample = (imgData, sampleId) => {

            // create the outermost row div element
            var row = document.createElement('div');
            row.setAttribute('class', 'row d-flex justify-content-center card-deck');
            row.setAttribute('id', 'card-deck');

            // create the profile card div element
            var sample = document.createElement('div');
            sample.setAttribute('class', 'profile-card-4 card text-center p-0 border-0 shadow');
            sample.setAttribute('id', sampleId);

            // create the close button for the profile card
            var closeButton = document.createElement('button');
            closeButton.setAttribute('class', 'close');
            closeButton.setAttribute('aria-label', 'Close');
            closeButton.innerHTML = '<span aria-hidden="true">&times;</span>';
            closeButton.addEventListener('click', function() {
                sample.remove(); // remove the profile card when close button is clicked
                samples = samples.filter(sample => sample.id !== sampleId);  // Remove the sample with the matching id from the array
                console.log(samples)
            });
            sample.appendChild(closeButton);

            // style the close button
            closeButton.style.position = 'absolute';
            closeButton.style.top = '0';
            closeButton.style.right = '0';
            closeButton.style.backgroundColor = 'transparent';
            closeButton.style.border = 'none';
            closeButton.style.outline = 'none';
            closeButton.style.color = 'white';
            closeButton.style.fontSize = '1.5rem';
            closeButton.style.cursor = 'pointer';
            closeButton.style.marginRight = '0.5rem';

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

            // create the second input field
            var col2 = document.createElement('div');
            col2.setAttribute('class', 'col-12 pb-3');
            var concentration = document.createElement('input');
            concentration.setAttribute('type', 'text');
            concentration.setAttribute('placeholder', 'Enter Concentration');
            concentration.setAttribute('id', 'concentration');
            col2.appendChild(concentration);

            // create the third input field
            var col3 = document.createElement('div');
            col3.setAttribute('class', 'col-12 pb-3');
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
            col3.appendChild(rating);

            // append the child elements to their respective parent elements
            row.appendChild(sample);
            sample.appendChild(img);
            sample.appendChild(div1);
            div1.appendChild(profileContent);
            profileContent.appendChild(row2);
            row2.appendChild(col1);
            row2.appendChild(col2);
            row2.appendChild(col3);

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

        let sampleCounter = 0;
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

            const sampleId = 'sample-' + sampleCounter;
            sampleCounter++
            const newsample = createSample(imgData, sampleId);  // Pass the id to createSample

            batchcontainer.appendChild(newsample);
            samples.push({id: sampleId, data: newsample});
            console.log(samples)
        });

        close.addEventListener('click', () => {

            const videoContainer = document.getElementById('videoContainer');
            const showVideo = document.getElementById('showVideo');
            var batchContainer = document.getElementById('batchContainer');
            videoContainer.style.cssText = 'display: none !important;';
            showVideo.style.display = 'block';
            batchContainer.style.marginTop = '30px';

            // Position the upload button
            // setFixedStyle(batchUpload, 'calc(0% + 25px)', 'calc(50% - 50px)', '20%', '15%');
        });

        show.addEventListener('click', () => {
            const videoContainer = document.getElementById('videoContainer');
            const showVideo = document.getElementById('showVideo');
            var batchContainer = document.getElementById('batchContainer');
            videoContainer.style.cssText = 'flex';
            showVideo.style.display = 'none';
            batchContainer.style.marginTop = '80px';
        });

        async function submitForm() {
            const csrf_token = getCsrfToken();

            const promises = [];
            const formData = new FormData();

            for (let i = 0; i < samples.length; i++) {

                const sampleElement = samples[i].data;

                const formulationInput = sampleElement.querySelector('#formulation');
                const concentrationInput = sampleElement.querySelector('#concentration');
                const ratingInput = sampleElement.querySelector('#rating');

                const formulationValue = formulationInput ? formulationInput.value || "Not provided" : "Not provided";
                const concentrationValue = concentrationInput ? concentrationInput.value || "Not provided" : "Not provided";
                const ratingValue = ratingInput ? ratingInput.value || "Not provided" : "Not provided";

                const imgSrc = sampleElement.querySelector('img').src;
                formData.append(`samples[${i}][img_data]`, imgSrc);
                formData.append(`samples[${i}][formulation]`, formulationValue);
                formData.append(`samples[${i}][concentration]`, concentrationValue);
                formData.append(`samples[${i}][rating]`, ratingValue);
            }

            // Create a new promise for each request
            const xhr = new XMLHttpRequest();
            xhr.open("POST", batchSubmitUrl, true)
            xhr.setRequestHeader('X-CSRFToken', csrf_token)

            xhr.onload = function () {
                if (xhr.status === 200) {
                    console.log("Success");
                    window.location.reload()
                } else {
                    console.error("Error");
                    window.location.reload()
                }
            };

            xhr.onerror = function () {
                console.error("Error");
            };

            await xhr.send(formData);

            // Reload the page after all uploads are complete
        }

        upload.addEventListener('click', () => {
            upload.style.display = 'none';
            const videoContainer = document.getElementById('videoContainer');
            videoContainer.style.cssText = 'display: none !important;';
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

            video.style = "initial";
            // Object.assign(videoContainer.style, { position: 'fixed', top: '50%', left: '50%', width: '50%', height: '50%', transform: 'translate(-50%, -50%)', overflow: 'hidden' });
            Object.assign(buttonWrapper.style, {
                position: 'fixed',
                bottom: '10px',
                left: '50%',
                width: '50%',
                height: '50px',
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
