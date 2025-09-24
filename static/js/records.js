document.addEventListener('DOMContentLoaded', function () {
    // DOM Elements
    const tableBody = document.getElementById('face-container');
    const allRows = Array.from(tableBody.querySelectorAll('.recognition-card'));
    const prevBtn = document.getElementById('prevBtn');
    const nextBtn = document.getElementById('nextBtn');
    const pageInfo = document.getElementById('pageInfo');
    const searchInput = document.getElementById('searchInput');
    const tabs = document.querySelectorAll('.tab');
    const rowsPerPage = 10;
    let currentPage = 1;
    let filteredRows = allRows.slice();

    // Incident Modal Elements
    const modal = document.getElementById('incidentModal');
    const openBtn = document.getElementById('addIncidentBtn');
    const closeBtn = modal.querySelector('.close-btn');
    const form = document.getElementById('incidentForm');
    const list = document.getElementById('incidentList');

    // Pagination & Filtering
    function updatePagination() {
        const totalPages = Math.max(1, Math.ceil(filteredRows.length / rowsPerPage));
        pageInfo.textContent = `Page ${currentPage} of ${totalPages}`;
        prevBtn.disabled = currentPage === 1;
        nextBtn.disabled = currentPage === totalPages || totalPages === 0;
    }

    function renderPage(page) {
        const start = (page - 1) * rowsPerPage;
        const end = start + rowsPerPage;
        
        // Hide all rows first
        allRows.forEach(row => row.style.display = 'none');
        
        // Show only filtered and paginated rows
        filteredRows.slice(start, end).forEach(row => row.style.display = '');
        
        updatePagination();
    }

    function applyFilters() {
        const term = searchInput.value.trim().toLowerCase();
        const activeTab = document.querySelector('.tab.active');
        const filter = activeTab ? activeTab.dataset.filter : 'all';
        
        filteredRows = allRows.filter(row => {
            const name = row.querySelector('.name').textContent.toLowerCase();
            const statusElement = row.querySelector('.value.recognized') || row.querySelector('.value');
            const status = statusElement ? statusElement.textContent.toLowerCase() : '';
            const direction = row.querySelector('.value') ? row.querySelector('.value').textContent.toLowerCase() : '';
            
            // Filter by status (recognized/unknown) or direction (entering/exiting)
            let statusMatch = true;
            if (filter === 'recognized' || filter === 'unknown') {
                statusMatch = status.includes(filter);
            } else if (filter === 'entering' || filter === 'exiting') {
                statusMatch = direction.includes(filter);
            }
            
            // Search term match
            const searchMatch = !term || name.includes(term);
            
            return statusMatch && searchMatch;
        });
        
        currentPage = 1;
        renderPage(currentPage);
    }

    // Tab clicks
    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            tabs.forEach(t => t.classList.remove('active'));
            tab.classList.add('active');
            applyFilters();
        });
    });

    // Search input
    searchInput.addEventListener('input', applyFilters);

    // Prev / Next
    prevBtn.addEventListener('click', () => {
        if (currentPage > 1) {
            currentPage--;
            renderPage(currentPage);
        }
    });
    
    nextBtn.addEventListener('click', () => {
        const totalPages = Math.ceil(filteredRows.length / rowsPerPage);
        if (currentPage < totalPages) {
            currentPage++;
            renderPage(currentPage);
        }
    });

    // INITIAL RENDER
    applyFilters();

    // ---- Incident Modal Logic ----
    // Show modal
    openBtn.addEventListener('click', () => {
        form.reset();
        modal.style.display = 'block';
    });

    // Hide modal
    closeBtn.addEventListener('click', () => {
        modal.style.display = 'none';
    });
    
    window.addEventListener('click', e => {
        if (e.target === modal) modal.style.display = 'none';
    });

    // Submit new incident
    form.addEventListener('submit', async e => {
        e.preventDefault();

        const data = {
            name: form.name.value.trim(),
            status: form.status.value,
            description: form.description.value.trim()
        };

        try {
            const resp = await fetch('/incidents/add', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(data)
            });

            if (!resp.ok) throw new Error('Save failed');

            const newInc = await resp.json();

            // Create new card element
            const card = document.createElement('li');
            card.innerHTML = `
                <div class="incident-info">
                    <strong>${newInc.name}</strong>
                    <p class="incident-description">${newInc.description || 'No description provided'}</p>
                </div>
                <div class="incident-meta">
                    <small>${newInc.timestamp.split(' ')[1]}</small>
                    <span class="badge ${newInc.status.toLowerCase()}">${newInc.status}</span>
                </div>
            `;

            // Add to list
            list.prepend(card);
            modal.style.display = 'none';
            form.reset();
        } catch (err) {
            alert('Failed to save incident. Please try again.');
            console.error(err);
        }
    });
});