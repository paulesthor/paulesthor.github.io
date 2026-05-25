// Script pour préserver le filtre actif lors du retour à la page projets
document.addEventListener('DOMContentLoaded', function () {
    // Récupérer le bouton retour
    const backButton = document.querySelector('a[href="Projet.html"]');

    if (backButton) {
        // Déterminer la catégorie du projet actuel
        const badge = document.querySelector('.project-badge');
        let category = 'all';

        if (badge) {
            if (badge.classList.contains('scolaire')) {
                category = 'scolaire';
            } else if (badge.classList.contains('personnel')) {
                category = 'perso';
            }
        }

        // Modifier le lien pour inclure le filtre
        if (category !== 'all') {
            backButton.href = `Projet.html?filter=${category}`;
        }
    }
});
