# Welcome to Generative AI Study Hub

<small>Last updated: {{ git_revision_date_localized }}</small>

This is your central hub for learning and referencing Generative AI concepts. 
Choose from the **Study Path** to go through structured topics or explore the **Reference Hub** for focused research and tooling insights.
<!-- Force MkDocs to include PWA assets in site/ output -->
[manifest]: manifest.webmanifest
[sw]: service-worker.js


<script>
  if ('serviceWorker' in navigator) {
    window.addEventListener('load', function () {
      navigator.serviceWorker.register(`${window.location.origin}/generative-ai-study-hub/service-worker.js`)
        .then(function (registration) {
          console.log('✅ Service Worker registered with scope:', registration.scope);
        })
        .catch(function (error) {
          console.log('❌ Service Worker registration failed:', error);
        });
    });
  }
</script>