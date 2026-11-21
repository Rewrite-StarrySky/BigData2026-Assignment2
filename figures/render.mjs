// Render Graphviz DOT files to SVG with the WebAssembly build of Graphviz (no system `dot` needed).
// usage: node render.mjs [--engine=dot|fdp|neato] file.dot ...
import { instance } from '@viz-js/viz';
import fs from 'fs';
const viz = await instance();
const engine = (process.argv.find(a => a.startsWith('--engine=')) || '--engine=dot').split('=')[1];
for (const f of process.argv.slice(2).filter(a => !a.startsWith('--'))) {
  const out = f.replace(/\.dot$/, '.svg');
  fs.writeFileSync(out, viz.renderString(fs.readFileSync(f, 'utf8'), { format: 'svg', engine }));
  console.log('wrote', out, `(${engine})`);
}
