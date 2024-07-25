document.getElementById('upload-form').addEventListener('submit', async function(event) {
    event.preventDefault();

    const formData = new FormData(this);
    const response = await fetch('/upload', {
        method: 'POST',
        body: formData
    });

    if (response.ok) {
        const result = await response.json();
        displayFeedback(result);
    } else {
        console.error('Error uploading files');
    }
});

function displayFeedback(data) {
    document.getElementById('feedback').style.display = 'block';
    
    // Accessing matched and missing skills directly
    const matchedSkills = data.matched_skills.join(', ');
    document.getElementById('matched-skills-list').textContent = matchedSkills || 'No matched skills';

    const missingSkills = data.missing_skills.join(', ');
    document.getElementById('missing-skills-list').textContent = missingSkills || 'No missing skills';

    // Recommendations
    document.getElementById('recommendations-text').textContent = data.recommendations || 'No recommendations';

    // Similarity Score
    document.getElementById('similarity-score').textContent = data.similarity_score ? `${data.similarity_score.toFixed(2)}` : 'No score available';

    // Notes
    const notesList = document.getElementById('notes-list');
    notesList.innerHTML = '';
    data.notes.forEach(note => {
        const li = document.createElement('li');
        li.textContent = note;
        notesList.appendChild(li);
    });
}
