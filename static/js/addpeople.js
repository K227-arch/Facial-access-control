// --- Element refs & config ---
const existingCount    = parseInt(document.getElementById('current-count').textContent || '0');
const minRequired      = parseInt(document.getElementById('drop-zone').dataset.minImages || '10');

const dropZone         = document.getElementById('drop-zone');
const fileInput        = document.getElementById('photos');
const previewContainer = document.getElementById('preview');
const countDisplay     = document.getElementById('current-count');
const actionBtns       = document.getElementById('action-buttons');
const trainBtn         = document.getElementById('train-btn');
const toastEl          = document.getElementById('toast');

// --- Helpers ---
function showToast(msg, type='info'){
  toastEl.className = `toast show toast-${type}`;
  toastEl.textContent = msg;
  setTimeout(()=> toastEl.className = 'toast', 3000);
}

function updateCount(total){
  countDisplay.textContent = total;
}

function updateButtons(total){
  // always show the wrapper
  actionBtns.style.display = 'flex';
  // only show Train if we've met the threshold
  trainBtn.style.display = (total >= minRequired) ? 'inline-block' : 'none';
}

function renderPreviews(files){
  previewContainer.innerHTML = '';
  files.forEach(file=>{
    const reader = new FileReader();
    reader.onload = e=>{
      const img = document.createElement('img');
      img.src   = e.target.result;
      img.classList.add('thumb');
      previewContainer.appendChild(img);
    };
    reader.readAsDataURL(file);
  });
}

// --- Event handlers ---
function onFilesChange(){
  const files = Array.from(fileInput.files);
  const total = existingCount + files.length;

  if(total < minRequired){
    showToast(`Need ${minRequired - total} more image(s) to train.`, 'warning');
  }

  updateCount(total);
  updateButtons(total);
  renderPreviews(files);
}

function bindDragDrop(){
  ['dragenter','dragover'].forEach(evt=>
    dropZone.addEventListener(evt, e=>{
      e.preventDefault();
      dropZone.classList.add('drag-over');
    })
  );
  ['dragleave','drop'].forEach(evt=>
    dropZone.addEventListener(evt, e=>{
      e.preventDefault();
      dropZone.classList.remove('drag-over');
      if(evt === 'drop'){
        fileInput.files = e.dataTransfer.files;
        onFilesChange();
      }
    })
  );
}

// --- Initialization ---

document.addEventListener('DOMContentLoaded', () => {
  updateCount(existingCount);
  updateButtons(existingCount);

  dropZone.addEventListener('click', () => fileInput.click());
  fileInput.addEventListener('change', onFilesChange);
  bindDragDrop();

  // Show server-side flash messages
  document.querySelectorAll('.flash-messages .alert').forEach(el => {
    const type = el.classList.contains('alert-danger') ? 'danger' :
                 el.classList.contains('alert-success') ? 'success' :
                 el.classList.contains('alert-warning') ? 'warning' : 'info';
    showToast(el.textContent, type);
  });
});







//


const peopleListEl = document.getElementById('peopleList');
const totalUsersEl = document.getElementById('totalUsers');
const totalUsersTextEl = document.getElementById('totalUsersText');
const searchInput = document.getElementById('searchInput');

let people = [];

// Fetch people list from the API
function fetchPeople() {
  fetch('/api/people')
    .then(res => res.json())
    .then(data => {
      people = data;
      applyFilters();
    })
    .catch(err => {
      console.error('Failed to fetch people:', err);
      peopleListEl.innerHTML = '<p class="error">Failed to load people data.</p>';
    });
}

// Render filtered people list
function renderPeople(list) {
  peopleListEl.innerHTML = '';
  list.forEach(person => {
    const card = document.createElement('div');
    card.className = 'person-card';

    const details = document.createElement('div');
    details.className = 'person-details';

    const info = document.createElement('div');
    info.className = 'person-info';
    info.innerHTML = `<i class="fa-solid fa-user"></i><div class="name">${person.name}</div>`;

    const btn = document.createElement('button');
    btn.className = 'delete-btn';
    btn.innerHTML = `<i class="fa-solid fa-trash-can"></i> Delete User`;
    btn.addEventListener('click', () => deleteUser(person.id));

    details.append(info, btn);

    const avatar = document.createElement('img');
    avatar.className = 'avatar';
    avatar.src = person.avatar;
    avatar.alt = person.name;

    card.append(details, avatar);
    peopleListEl.append(card);
  });

  totalUsersEl.textContent = list.length;
  totalUsersTextEl.textContent = list.length;
}

// Apply search filter
function applyFilters() {
  const q = searchInput.value.toLowerCase();
  const filtered = people.filter(p => p.name.toLowerCase().includes(q));
  renderPeople(filtered);
}

// Handle user deletion
function deleteUser(id) {
  if (confirm(`Delete user "${id}"? This action cannot be undone.`)) {
    fetch(`/api/people/${encodeURIComponent(id)}`, {
      method: 'DELETE'
    })
      .then(res => {
        if (res.ok) {
          fetchPeople();
        } else {
          alert('Failed to delete user');
        }
      });
  }
}

// Event listeners
searchInput.addEventListener('input', applyFilters);

// Initialize
fetchPeople();