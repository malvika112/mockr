document.addEventListener('DOMContentLoaded', function() {
    var audio = document.getElementById('questionAudio');
    function playAudio(base64Audio) {
        var audioSource = 'data:audio/mp3;base64,' + base64Audio;
        audio.src = audioSource;
        audio.play();
    }

    // Function to update the transcript
    window.updateTranscript = function(transcript) {
        document.getElementById('transcript').textContent = transcript;
    };

    // Function to handle new questions and audio from the server
    window.updateQuestionAudio = function(question, base64Audio) {
        document.querySelector('p').textContent = 'Question: ' + question;
        playAudio(base64Audio);
    };

    // Add event listener for the start interview button
    const startInterviewButton = document.getElementById('startInterviewButton');
    if (startInterviewButton) {
        // Check if question_audio_base64 is available in the template context
        if (typeof question_audio_base64 !== 'undefined' && question_audio_base64) {
            playAudio(question_audio_base64);
        }

        startInterviewButton.addEventListener('click', function() {
            fetch('/api/interview_action', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ action: 'start_interview' })
            })
            .then(response => response.json())
            .then(data => {
                if (data.status === 'success') {
                    window.updateQuestionAudio(data.question, data.question_audio);
                } else {
                    alert('Error starting interview: ' + data.message);
                }
            })
            .catch(error => {
                console.error('Error:', error);
                alert('Error starting interview: ' + error);
            });
        });
    }

    const nextQuestionButton = document.getElementById('nextQuestionButton');
    if (nextQuestionButton) {
        nextQuestionButton.addEventListener('click', function() {
            // Get the answer from the textarea
            const answer = document.getElementById('answer').value;

            fetch('/api/interview_action', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ action: 'submit_answer', user_answer: answer })
            })
            .then(response => response.json())
            .then(data => {
                if (data.status === 'success') {
                    window.updateQuestionAudio(data.next_question, data.next_question_audio);
                } else {
                    alert('Error getting next question: ' + data.message);
                }
                // Fetch the transcript from the /process_audio_chunk route
                fetch('/process_audio_chunk', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ text_chunk: '' })
                })
                .then(response => response.json())
                .then(data => {
                    if (data.status === 'analyzed') {
                        window.updateTranscript(data.transcript);
                    } else {
                        console.error('Error getting transcript: ' + data.message);
                    }
                })
                .catch(error => {
                    console.error('Error:', error);
                    alert('Error getting transcript: ' + error);
                });
            })
            .catch(error => {
                console.error('Error:', error);
                alert('Error getting next question: ' + error);
            });
        });
    }

    const endInterviewButton = document.getElementById('endInterviewButton');
    if (endInterviewButton) {
        endInterviewButton.addEventListener('click', function() {
            fetch('/api/interview_action', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ action: 'end_interview' })
            })
            .then(response => response.json())
            .then(data => {
                if (data.redirect) {
                    window.location.href = data.url;
                } else {
                    alert('Error ending interview: ' + data.message);
                }
            })
            .catch(error => {
                console.error('Error:', error);
                alert('Error ending interview: ' + error);
            });
        });
    }
});
