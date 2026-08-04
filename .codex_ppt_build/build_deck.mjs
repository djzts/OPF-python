import fs from 'node:fs/promises';
import { FileBlob, PresentationFile } from '@oai/artifact-tool';

const tmp = 'E:/ZZQ_python_script/OPF_data/OPF-python/.codex_ppt_build';
const out = 'E:/OneDrive - Brookhaven National Laboratory/Documents/ZZQ_bnl_files/Weekly Report/QHD_LALM_ACOPF_5slides.pptx';
const asset = 'E:/ZZQ_python_script/OPF_data/OPF-python/output';
const p = await PresentationFile.importPptx(await FileBlob.load(`${tmp}/template-starter.pptx`));

const C = { blue:'#0070C0', green:'#2E7D17', dark:'#171717', gray:'#666666', light:'#E9EEF3', red:'#A33A22' };
function box(slide, name, text, x,y,w,h,size=20,color=C.dark,bold=false,align='left') {
  const s=slide.shapes.add({geometry:'textbox',name,position:{left:x,top:y,width:w,height:h},fill:'none',line:{style:'solid',fill:'none',width:0}});
  s.text=text; s.text.style={fontFamily:'Aptos',fontSize:size,color,bold,alignment:align,verticalAlignment:'middle'}; return s;
}
function title(slide, t, kicker) {
  box(slide,'section-kicker',kicker.toUpperCase(),64,28,400,24,13,C.blue,true);
  box(slide,'slide-title',t,64,52,1152,58,32,C.dark,true);
  box(slide,'rule','━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━',64,110,1152,18,10,C.light,false);
}
function footer(slide,n){ box(slide,'footer-left','QHD–LALM–SB for ACOPF  |  repository results',64,684,720,18,11,'#777777'); box(slide,'footer-page',String(n).padStart(2,'0'),1150,684,66,18,11,C.gray,true,'right'); }
async function addImg(slide,path,x,y,w,h,alt){ const b=await fs.readFile(path); slide.images.add({blob:b.buffer.slice(b.byteOffset,b.byteOffset+b.byteLength),contentType:'image/png',alt,fit:'contain',position:{left:x,top:y,width:w,height:h}}); }
function shiftScale(slide, sx, sy, dx, dy){ for(const s of slide.shapes.items){ const q=s.position; if(!q) continue; s.position={left:q.left*sx+dx,top:q.top*sy+dy,width:q.width*sx,height:q.height*sy}; } }

// 1 — Motivation
{
 const s=p.slides.getItem(0); shiftScale(s,.68,.68,650,155); title(s,'Why combine global search with local feasibility restoration?','Motivation');
 box(s,'thesis','ACOPF is smooth—but strongly nonconvex.',64,145,525,48,25,C.dark,true);
 box(s,'motivation-body','• Classical solvers converge rapidly near a good basin, but can depend on initialization and active-set choices.\n• QHD/SB searches a bounded lattice and can explore multiple candidate basins without requiring a single local trajectory.\n• Coarse lattice resolution leaves a feasibility floor; a classical refinement step is therefore not optional if the target is |h|max ≤ 10⁻⁵.',64,205,555,260,19,C.dark);
 box(s,'key-question','Research question',64,495,180,28,16,C.blue,true);
 box(s,'question','Can a QHD-guided outer loop locate useful ACOPF regions, while a local constrained update converts those candidates into accurate feasible solutions?',64,525,535,92,20,C.dark,true);
 box(s,'diagram-caption','Search broadly in the current box; restore feasibility locally.',720,590,450,36,16,C.gray,false,'center'); footer(s,1);
}

// 2 — Idea
{
 const s=p.slides.getItem(1); shiftScale(s,.72,.72,70,145); title(s,'Localize, refine, and repeat around the best feasible candidate','Core idea');
 box(s,'idea-1','1  QHD/SB coarse search',690,155,430,34,21,C.blue,true);
 box(s,'idea-1b','Sample a finite lattice inside the current variable bounds and return one or more low-energy candidates.',690,192,455,72,17,C.dark);
 box(s,'idea-2','2  Classical refinement',690,285,430,34,21,C.green,true);
 box(s,'idea-2b','Use TNC/SQP/interior-point-style local updates to reduce equality violations and recover a physically meaningful operating point.',690,322,455,88,17,C.dark);
 box(s,'idea-3','3  Bound localization',690,430,430,34,21,C.red,true);
 box(s,'idea-3b','Center the next box on the selected candidate, shrink the range, increase effective resolution, and update LALM penalty/multiplier terms.',690,467,455,88,17,C.dark);
 box(s,'idea-callout','Multi-beam = exploration insurance; single-beam = lower-cost exploitation.',690,585,455,48,18,C.dark,true); footer(s,2);
}

// 3 — Algorithm
{
 const s=p.slides.getItem(2);
 for(const sh of s.shapes.items){ const t=String(sh.text ?? ''); if((sh.position?.left ?? 0)>700 || /Repeat LALM|Stop when|Converged solution|Classically refined|QHD coarse|Localized QHD/.test(t)) sh.position={left:-2000,top:-2000,width:1,height:1}; }
 shiftScale(s,.78,.78,-180,95); title(s,'The solver alternates discrete search, continuous repair, and LALM updates','Algorithm / method');
 box(s,'method-steps','Initialize x⁰, bounds B⁰, α, ρ, μ\n\n1. Linearize / form the LALM subproblem\n2. Encode the bounded subproblem on a lattice\n3. Solve with QHD-SB; retain 1 or K beams\n4. Refine each candidate continuously\n5. Rank by feasibility first, objective second\n6. Update μ, ρ, α and shrink bounds\n7. Stop on residual, step, or iteration limits',650,150,500,330,18,C.dark);
 box(s,'selection-rule','Selection rule',650,500,150,26,16,C.blue,true);
 box(s,'selection-text','Reject deceptively low objectives from infeasible points. Compare objective quality only after attaching L₂(h) and max|h|.',650,530,500,70,18,C.dark,true);
 box(s,'implementation','Implemented across 2-, 3-, 5-, 9- and 14-bus scripts with coarse-only, single-beam, multi-beam and post-refine logging paths.',650,620,500,42,15,C.gray); footer(s,3);
}

// 4 — Results
{
 const s=p.slides.getItem(3); for(const sh of s.shapes.items){ sh.position={left:-2000,top:-2000,width:1,height:1}; } for(const tb of s.tables.items){ tb.position={left:-2000,top:-2000,width:1,height:1}; } title(s,'Refinement removes 81–99.8% of coarse residual across tested cases','Results: 2 / 3 / 5 / 9 bus');
 await addImg(s,`${asset}/coarse_vs_post_refine_improvement.png`,55,145,575,410,'Coarse versus post-refine residual improvement');
 await addImg(s,`${asset}/best_l2_residual_by_experiment.png`,650,145,570,410,'Best L2 residual by experiment');
 box(s,'results-strip','Best latest-log residual',64,575,215,24,15,C.blue,true);
 box(s,'results-values','2-bus  9.22×10⁻⁶   |   3-bus  2.39×10⁻⁶   |   5-bus  1.53×10⁻⁵   |   9-bus  1.40×10⁻⁴',64,604,1080,34,20,C.dark,true);
 box(s,'result-takeaway','Objective gaps stay small at these selected points (≈ −0.006% to +0.059%), but feasibility—not objective alone—remains the governing acceptance test.',64,646,1080,30,15,C.gray); footer(s,4);
}

// 5 — Analysis
{
 const s=p.slides.getItem(4); for(const sh of s.shapes.items){ sh.position={left:-2000,top:-2000,width:1,height:1}; } title(s,'The hybrid architecture works; beam policy and stopping logic now dominate','Analysis & next decisions');
 await addImg(s,`${asset}/3bus_tnc_vs_qhdsb_convergence.png`,64,150,610,400,'3-bus TNC versus QHD-SB convergence');
 box(s,'a1','What the evidence says',710,150,430,28,20,C.blue,true);
 box(s,'a1b','• Multi-beam wins on 2- and 3-bus residuals.\n• Single-beam wins on 5- and 9-bus residuals.\n• Best-residual iterates can be better than the final iterate; retain and return the incumbent.\n• 9-bus remains above the 10⁻⁵ target, indicating scaling and conditioning limits.',710,188,450,205,17,C.dark);
 box(s,'a2','Recommended changes',710,420,430,28,20,C.green,true);
 box(s,'a2b','1. Make beam width adaptive to candidate diversity and recent residual progress.\n2. Trigger refinement earlier when coarse progress plateaus.\n3. Rank lexicographically: max|h| → L₂(h) → objective.\n4. Normalize runtime comparisons across hardware/backends.\n5. Use SQP/interior-point as deterministic validation references.',710,458,450,178,17,C.dark);
 box(s,'close','Bottom line: QHD is most valuable as a basin-discovery mechanism; continuous refinement is what delivers ACOPF-grade feasibility.',64,585,610,62,19,C.dark,true); footer(s,5);
}

await fs.mkdir(`${tmp}/final-render`,{recursive:true});
for(let i=0;i<p.slides.items.length;i++){
 const s=p.slides.getItem(i); const png=await p.export({slide:s,format:'png',scale:1}); await fs.writeFile(`${tmp}/final-render/slide-${i+1}.png`,new Uint8Array(await png.arrayBuffer()));
 const lay=await s.export({format:'layout'}); await fs.writeFile(`${tmp}/final-render/slide-${i+1}.layout.json`,await lay.text());
}
const deck=await PresentationFile.exportPptx(p); await deck.save(out);
console.log(out);
