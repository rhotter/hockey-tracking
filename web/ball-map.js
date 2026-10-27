/* Display positions only: preserves the tracker's floor observations unchanged. */
((root) => {
  'use strict';
  const inside = p => Array.isArray(p) && p.length === 2 && p.every(v => Number.isFinite(v) && v >= 0 && v <= 1);
  function position(estimate, row, bounds) {
    if (!estimate || estimate.state === 'unknown' || !inside(estimate.pixel)) return {xy:null, method:'unresolved'};
    if (!row?.camera_valid) return {xy:null, method:'camera_unavailable'};
    if (inside(estimate.xy)) return {xy:estimate.xy, method:'floor'};

    // A held ball follows its identified carrier's ground position when available.
    // Do not assign it to an unrelated nearby player or infer a new identity.
    if (estimate.review_support === 'held' && estimate.carrier_track_id) {
      const id = String(estimate.carrier_track_id).replace(/^T/, '');
      const carrier = row.objects?.find(o => String(o.track_id) === id);
      if (inside(carrier?.xy)) return {xy:carrier.xy, method:'carrier'};
    }

    // The camera ray's intersection with the floor is a schematic fallback,
    // not a recovered 3D location for a ball above the floor.
    const h = row.image_to_floor;
    if (!h || !bounds || bounds.length !== 4) return {xy:null, method:'camera_unavailable'};
    const p = [estimate.pixel[0]*1920, estimate.pixel[1]*1080, 1];
    const q = h.map(r => r.reduce((sum, v, i) => sum + v*p[i], 0));
    if (!q.every(Number.isFinite) || Math.abs(q[2]) < 1e-9) return {xy:null, method:'outside_rink'};
    const [lo,hi,far,near] = bounds;
    const xy = [(q[0]/q[2]-lo)/(hi-lo), (q[1]/q[2]-far)/(near-far)];
    if (!inside(xy)) return {xy:null, method:'outside_rink'};
    return {xy, method:estimate.review_support === 'floor' ? 'floor' : 'floor_approximation'};
  }
  const api = {position};
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.SportScansBallMap = api;
})(globalThis);
