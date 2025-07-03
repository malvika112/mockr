document.addEventListener('DOMContentLoaded', function() {
    const startButton = document.getElementById('start_button');
    const stopButton = document.getElementById('stop_button');
    const listenButton = document.getElementById('listen_button');
    const nextButton = document.getElementById('next_button');
    const endButton = document.getElementById('end_button');
    const answerTextarea = document.getElementById('answer');
    const submitButton = document.getElementById('submit_button');

    let recognition;
    let isRecognizing = false;

    // Function to speak the question
    function speak(text) {
        const synth = window.speechSynthesis;
        const utterance = new SpeechSynthesisUtterance(text);
        synth.speak(utterance);
    }

    // Function to automatically read the question
    function autoSpeakQuestion() {
        const questionText = document.querySelector('#question').textContent;
        speak(questionText);
    }

    // Call autoSpeakQuestion when the page loads
    autoSpeakQuestion();

    listenButton.addEventListener('click', () => {
        const questionText = document.querySelector('#question').textContent;
        speak(questionText);
    });

    startButton.addEventListener('click', () => {
        // Check if the SpeechRecognition API is supported
        if ('SpeechRecognition' in window || 'webkitSpeechRecognition' in window) {
            // Use SpeechRecognition API
            recognition = new (window.SpeechRecognition || window.webkitSpeechRecognition)();

            recognition.continuous = true;
            recognition.interimResults = false;
            recognition.lang = 'en-US';

            recognition.onstart = () => {
                console.log('Speech recognition started');
                answerTextarea.value = ''; // Clear previous answer
                startButton.disabled = true;
                stopButton.disabled = false;
                isRecognizing = true;
            };

            recognition.onresult = (event) => {
                const transcript = Array.from(event.results)
                    .map(result => result[0])
                    .map(result => result.transcript)
                    .join('');

                answerTextarea.value = transcript;
            };

            recognition.onerror = (event) => {
                console.error('Speech recognition error:', event.error);
                startButton.disabled = false;
                stopButton.disabled = true;
                isRecognizing = false;
            };

            recognition.onend = () => {
                console.log('Speech recognition ended');
                startButton.disabled = false;
                stopButton.disabled = true;
                isRecognizing = false;
            };

            recognition.start();
        } else {
            console.log('Speech Recognition API is not supported in this browser.');
            answerTextarea.placeholder = 'Speech Recognition API is not supported in this browser.';
            startButton.disabled = true;
            stopButton.disabled = true;
        }
    });

    stopButton.addEventListener('click', () => {
        if (recognition && isRecognizing) {
            recognition.stop();
            startButton.disabled = false;
            stopButton.disabled = true;
            isRecognizing = false;
        }
    });

    nextButton.addEventListener('click', () => {
         // Send the transcribed text to the server
        const answer = answerTextarea.value;
        const formData = new FormData();
        formData.append('answer', answer);

        fetch('/submit_answer', {
            method: 'POST',
            body: formData
        })
        .then(response => {
            if (response.redirected) {
                window.location.href = response.url; // Follow the redirect
            } else {
                console.log("Answer submitted successfully, but no redirect.");
            }
        })
        .catch(error => {
            console.error('Error:', error);
        });
    });

    endButton.addEventListener('click', () => {
        window.location.href = '/results';
    });

    submitButton.addEventListener('click', () => {
        // Send the transcribed text to the server
        const answer = answerTextarea.value;
        const formData = new FormData();
        formData.append('answer', answer);

        fetch('/submit_answer', {
            method: 'POST',
            body: formData
        })
        .then(response => {
            if (response.redirected) {
                window.location.href = response.url; // Follow the redirect
            } else {
                console.log("Answer submitted successfully, but no redirect.");
            }
        })
        .catch(error => {
            console.error('Error:', error);
        });
    });
});
