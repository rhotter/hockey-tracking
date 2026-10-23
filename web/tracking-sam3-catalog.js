(() => {
  const clips = ['pittsburgh-0600', 'pittsburgh-1800', 'marylandwomen-0600', 'marylandwomen-1800'];
  for (const clip of clips) {
    const spec = window.SportScansTrackingCatalog[clip];
    spec.methods.push('sam3-native', 'sam31-native');
    spec.methodLabels['sam3-native'] = 'SAM3 native · person prompt';
    spec.methodLabels['sam31-native'] = 'SAM 3.1 Object Multiplex · person prompt';
  }
})();
