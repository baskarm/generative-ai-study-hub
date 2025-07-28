<script>
  if ('serviceWorker' in navigator) {
    navigator.serviceWorker.register('service-worker.js', { scope: './' })
      .then(reg => console.log('✅ ServiceWorker registered:', reg))
      .catch(err => console.error('❌ ServiceWorker registration failed:', err));
  }
</script>