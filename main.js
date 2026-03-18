import * as THREE from 'three';
import { VRButton } from 'three/examples/jsm/webxr/VRButton.js';

// ─── Math helpers ─────────────────────────────────────────────────────────────

function det3(m) {
  return (
    m[0][0]*(m[1][1]*m[2][2]-m[1][2]*m[2][1])
   -m[0][1]*(m[1][0]*m[2][2]-m[1][2]*m[2][0])
   +m[0][2]*(m[1][0]*m[2][1]-m[1][1]*m[2][0])
  );
}

function mul(A, B) {
  const C = [[0,0,0],[0,0,0],[0,0,0]];
  for (let i=0;i<3;i++)
    for (let j=0;j<3;j++)
      for (let k=0;k<3;k++)
        C[i][j] += A[i][k]*B[k][j];
  return C;
}

function mulVec(M, v) {
  return [
    M[0][0]*v[0]+M[0][1]*v[1]+M[0][2]*v[2],
    M[1][0]*v[0]+M[1][1]*v[1]+M[1][2]*v[2],
    M[2][0]*v[0]+M[2][1]*v[1]+M[2][2]*v[2],
  ];
}

function rotAxisAngle(axis, angle) {
  const norm = Math.sqrt(axis[0]**2+axis[1]**2+axis[2]**2);
  if (norm < 1e-12 || Math.abs(angle) < 1e-12) return [[1,0,0],[0,1,0],[0,0,1]];
  const [x,y,z] = axis.map(a=>a/norm);
  const c=Math.cos(angle), s=Math.sin(angle), C=1-c;
  return [
    [c+x*x*C,   x*y*C-z*s, x*z*C+y*s],
    [y*x*C+z*s, c+y*y*C,   y*z*C-x*s],
    [z*x*C-y*s, z*y*C+x*s, c+z*z*C  ],
  ];
}

function axisAngle(R) {
  const tr = Math.max(-1, Math.min(3, R[0][0]+R[1][1]+R[2][2]));
  const theta = Math.acos(Math.max(-1, Math.min(1, (tr-1)/2)));
  if (Math.abs(theta) < 1e-8) return { axis:[0,0,1], angle:0 };
  const rx=R[2][1]-R[1][2], ry=R[0][2]-R[2][0], rz=R[1][0]-R[0][1];
  const n=Math.sqrt(rx*rx+ry*ry+rz*rz);
  return { axis: n<1e-12?[1,0,0]:[rx/n,ry/n,rz/n], angle: theta };
}

// ─── Jacobi SVD (3×3) ─────────────────────────────────────────────────────────

function transpose3(A) {
  return [[A[0][0],A[1][0],A[2][0]],[A[0][1],A[1][1],A[2][1]],[A[0][2],A[1][2],A[2][2]]];
}

// Jacobi eigendecomposition of a 3×3 symmetric matrix.
// Returns { vals:[3], vecs:[[3],[3],[3]] } (vecs columns = eigenvectors).
function jacobiEig3(S) {
  let A = S.map(r=>[...r]);
  let V = [[1,0,0],[0,1,0],[0,0,1]];
  for (let iter=0; iter<60; iter++) {
    let p=0,q=1;
    for (let i=0;i<3;i++) for (let j=i+1;j<3;j++)
      if (Math.abs(A[i][j])>Math.abs(A[p][q])) { p=i; q=j; }
    if (Math.abs(A[p][q])<1e-14) break;
    const th=0.5*Math.atan2(2*A[p][q],A[q][q]-A[p][p]);
    const c=Math.cos(th), s=Math.sin(th);
    const newA=A.map(r=>[...r]);
    for (let i=0;i<3;i++) {
      const ip=A[i][p]*c+A[i][q]*s, iq=-A[i][p]*s+A[i][q]*c;
      newA[i][p]=ip; newA[p][i]=ip; newA[i][q]=iq; newA[q][i]=iq;
    }
    newA[p][p]=A[p][p]*c*c+2*A[p][q]*c*s+A[q][q]*s*s;
    newA[q][q]=A[p][p]*s*s-2*A[p][q]*c*s+A[q][q]*c*c;
    newA[p][q]=0; newA[q][p]=0;
    A=newA;
    for (let i=0;i<3;i++) {
      const vip=V[i][p]*c+V[i][q]*s, viq=-V[i][p]*s+V[i][q]*c;
      V[i][p]=vip; V[i][q]=viq;
    }
  }
  return { vals:[A[0][0],A[1][1],A[2][2]], vecs:V };
}

function svd3(A) {
  const ATA=mul(transpose3(A),A);
  const {vals,vecs}=jacobiEig3(ATA);
  const idx=[0,1,2].sort((a,b)=>vals[b]-vals[a]);
  const sigmas=idx.map(i=>Math.sqrt(Math.max(0,vals[i])));
  // V: columns are right singular vectors
  const V=[[0,0,0],[0,0,0],[0,0,0]];
  for(let j=0;j<3;j++) for(let i=0;i<3;i++) V[i][j]=vecs[i][idx[j]];
  // U: U_j = A*v_j / sigma_j
  const U=[[0,0,0],[0,0,0],[0,0,0]];
  for(let j=0;j<3;j++){
    const vj=[V[0][j],V[1][j],V[2][j]];
    const Avj=mulVec(A,vj);
    const s=sigmas[j];
    if(s>1e-10) for(let i=0;i<3;i++) U[i][j]=Avj[i]/s;
    else        for(let i=0;i<3;i++) U[i][j]=(i===j)?1:0;
  }
  return {U, S:sigmas, V};
}

// ─── Rotational SVD ───────────────────────────────────────────────────────────

function makeRotationalSVD(A) {
  const {U,S,V}=svd3(A);
  const Rf=[[1,0,0],[0,1,0],[0,0,-1]];
  let Sm=[[S[0],0,0],[0,S[1],0],[0,0,S[2]]];
  const dU=det3(U), dV=det3(V);
  if (dU<0&&dV<0)  return {U:mul(U,Rf), Sigma:Sm,        V:mul(V,Rf)};
  if (dU<0)        return {U:mul(U,Rf), Sigma:mul(Rf,Sm), V};
  if (dV<0)        return {U,           Sigma:mul(Sm,Rf), V:mul(V,Rf)};
  return {U, Sigma:Sm, V};
}

function svdPath(t, { U, Sigma, V }) {
  const { axis:aV, angle:tV } = axisAngle(V);
  const { axis:aU, angle:tU } = axisAngle(U);
  const s1=Sigma[0][0], s2=Sigma[1][1], s3=Sigma[2][2];
  if (t <= 1.0) {
    return rotAxisAngle(aV, t*tV);
  } else if (t <= 2.0) {
    const a=t-1;
    return mul(rotAxisAngle(aV,tV), [[1+a*(s1-1),0,0],[0,1+a*(s2-1),0],[0,0,1+a*(s3-1)]]);
  } else {
    return mul(mul(rotAxisAngle(aV,tV), Sigma), rotAxisAngle(aU, -(t-2)*tU));
  }
}

// ─── Preset matrices ──────────────────────────────────────────────────────────

function buildRotScale() {
  const c=Math.cos(Math.PI/5), s=Math.sin(Math.PI/5);
  return mul([[c,-s,0],[s,c,0],[0,0,1]], [[1.8,0,0],[0,0.7,0],[0,0,1.2]]);
}

const PRESETS = [
  { name:'Symmetric',     A:[[1.0,0.2,0.0],[0.2,1.2,0.1],[0.0,0.1,0.8]] },
  { name:'Rotation+Scale',A:buildRotScale() },
  { name:'Shear+Scale',   A:[[1.5,0.8,0.2],[0.0,1.0,0.5],[0.1,0.0,0.6]] },
];

// ─── Point cloud (seeded LCG) ─────────────────────────────────────────────────

function generatePoints(n, seed) {
  let s = seed >>> 0;
  const rng = () => { s=(s*1664525+1013904223)&0xffffffff; return (s>>>0)/0xffffffff; };
  const randn = () => Math.sqrt(-2*Math.log(rng()||1e-10))*Math.cos(2*Math.PI*rng());
  // Cholesky of cov [[1,.4,.2],[.4,1.2,.3],[.2,.3,.8]]
  const L00=1, L10=0.4, L11=Math.sqrt(1.2-0.16);
  const L20=0.2, L21=(0.3-0.4*0.2)/L11, L22=Math.sqrt(Math.max(0,0.8-0.04-L21**2));
  const pts=[];
  for(let i=0;i<n;i++){
    const z=[randn(),randn(),randn()];
    pts.push([L00*z[0], L10*z[0]+L11*z[1], L20*z[0]+L21*z[1]+L22*z[2]]);
  }
  return pts;
}

// ─── Cube corners ─────────────────────────────────────────────────────────────

function buildCube(pts) {
  let xn=Infinity,yn=Infinity,zn=Infinity,xx=-Infinity,yx=-Infinity,zx=-Infinity;
  for(const p of pts){ if(p[0]<xn)xn=p[0];if(p[0]>xx)xx=p[0];if(p[1]<yn)yn=p[1];if(p[1]>yx)yx=p[1];if(p[2]<zn)zn=p[2];if(p[2]>zx)zx=p[2]; }
  const cx=(xn+xx)/2,cy=(yn+yx)/2,cz=(zn+zx)/2;
  const h=0.5*Math.max(xx-xn,yx-yn,zx-zn)*1.3+1e-6;
  return [
    [cx-h,cy-h,cz-h],[cx+h,cy-h,cz-h],[cx+h,cy+h,cz-h],[cx-h,cy+h,cz-h],
    [cx-h,cy-h,cz+h],[cx+h,cy-h,cz+h],[cx+h,cy+h,cz+h],[cx-h,cy+h,cz+h],
  ];
}

const EDGES=[[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]];

function applyM(pts, M) { return pts.map(p=>mulVec(M,p)); }

function cubePositions(corners) {
  const pos=[];
  for(const [i,j] of EDGES) pos.push(...corners[i],...corners[j]);
  return pos;
}

function dispPositions(orig, trans) {
  const pos=[];
  for(let i=0;i<orig.length;i++) pos.push(...orig[i],...trans[i]);
  return pos;
}

// ─── Three.js setup ───────────────────────────────────────────────────────────

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x111118);
scene.fog = new THREE.Fog(0x111118, 15, 40);

const camera = new THREE.PerspectiveCamera(70, window.innerWidth/window.innerHeight, 0.01, 100);
camera.position.set(0, 1.6, 5);

const renderer = new THREE.WebGLRenderer({ antialias:true });
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.xr.enabled = true;
document.body.appendChild(renderer.domElement);
document.body.appendChild(VRButton.createButton(renderer));

scene.add(new THREE.HemisphereLight(0xffffff,0x223344,1.2));
const dl=new THREE.DirectionalLight(0xffffff,0.7); dl.position.set(5,8,5); scene.add(dl);

// ─── HUD ──────────────────────────────────────────────────────────────────────

const hud=document.createElement('div');
Object.assign(hud.style,{
  position:'absolute',top:'12px',left:'12px',color:'#fff',
  fontFamily:'monospace',fontSize:'15px',background:'rgba(0,0,0,0.6)',
  padding:'10px 16px',borderRadius:'8px',lineHeight:'1.8',pointerEvents:'none',
});
document.body.appendChild(hud);

function stageName(t) {
  if(t<0.01) return 'Identity (I)';
  if(t<=1.0) return 'Stage 1: V rotation';
  if(t<=2.0) return 'Stage 2: Σ scaling';
  return 'Stage 3: U rotation → A';
}
function updateHUD() {
  hud.innerHTML=
    `Matrix: <b>${PRESETS[presetIdx].name}</b> &nbsp;[keys 1/2/3 | grip]<br>`+
    `t = <b>${tParam.toFixed(2)}</b> / 3.00<br>`+
    `<b>${stageName(tParam)}</b><br>`+
    `<span style="color:#aaa;font-size:12px">Hold ← → to scrub | hold triggers in VR</span>`;
}

// ─── Scene graph ──────────────────────────────────────────────────────────────

const root=new THREE.Group();
root.scale.setScalar(0.55);
root.position.set(0,1.2,-1.8);
scene.add(root);

const sphereGeo=new THREE.SphereGeometry(0.05,8,6);
const matOrig=new THREE.MeshLambertMaterial({color:0x4488ff});
const matTrans=new THREE.MeshLambertMaterial({color:0xff7722});
const matCubeO=new THREE.LineBasicMaterial({color:0x888888});
const matCubeT=new THREE.LineBasicMaterial({color:0xff3333,linewidth:2});
const matDisp=new THREE.LineBasicMaterial({color:0x6699bb,transparent:true,opacity:0.4});
const matAxisV=new THREE.LineDashedMaterial({color:0x9933ff,dashSize:0.12,gapSize:0.06,transparent:true,opacity:1});
const matAxisU=new THREE.LineDashedMaterial({color:0x00cccc,dashSize:0.12,gapSize:0.06,transparent:true,opacity:0});
const EIG_MATS=[
  new THREE.LineBasicMaterial({color:0x44dd66}),
  new THREE.LineBasicMaterial({color:0xffaa33}),
  new THREE.LineBasicMaterial({color:0xffdd44}),
];

let origGroup, transGroup, cubeOLine, cubeTLine, dispLine, axisVLine, axisULine, eigGroup;

function makeLineSegs(positions, mat) {
  const geo=new THREE.BufferGeometry();
  geo.setAttribute('position',new THREE.Float32BufferAttribute(positions,3));
  return new THREE.LineSegments(geo,mat);
}

function updateLineSegs(obj, positions) {
  obj.geometry.setAttribute('position',new THREE.Float32BufferAttribute(positions,3));
  obj.geometry.attributes.position.needsUpdate=true;
}

function makeAxisLine(ax, len, mat) {
  const n=Math.sqrt(ax[0]**2+ax[1]**2+ax[2]**2);
  if(n<1e-12) return null;
  const a=ax.map(x=>x/n);
  const geo=new THREE.BufferGeometry().setFromPoints([
    new THREE.Vector3(-a[0]*len,-a[1]*len,-a[2]*len),
    new THREE.Vector3( a[0]*len, a[1]*len, a[2]*len),
  ]);
  const line=new THREE.Line(geo,mat);
  line.computeLineDistances();
  return line;
}

// ─── State ────────────────────────────────────────────────────────────────────

let tParam=0, presetIdx=0, currentSVD=null, points=[], cubeCorners=[];

function rebuildScene() {
  root.clear();
  root.add(new THREE.AxesHelper(1.8));

  const A=PRESETS[presetIdx].A;
  currentSVD=makeRotationalSVD(A);
  points=generatePoints(30,42);
  cubeCorners=buildCube(points);

  // Original points
  origGroup=new THREE.Group();
  for(const p of points){ const m=new THREE.Mesh(sphereGeo,matOrig); m.position.set(...p); origGroup.add(m); }
  root.add(origGroup);

  // Transformed points
  transGroup=new THREE.Group();
  for(const p of points){ const m=new THREE.Mesh(sphereGeo,matTrans); m.position.set(...p); transGroup.add(m); }
  root.add(transGroup);

  // Cubes
  cubeOLine=makeLineSegs(cubePositions(cubeCorners),matCubeO); root.add(cubeOLine);
  cubeTLine=makeLineSegs(cubePositions(cubeCorners),matCubeT); root.add(cubeTLine);

  // Displacement lines
  dispLine=makeLineSegs(dispPositions(points,points),matDisp); root.add(dispLine);

  // Rotation axes
  const {axis:aV}=axisAngle(currentSVD.V);
  const {axis:aU}=axisAngle(currentSVD.U);
  axisVLine=makeAxisLine(aV,3,matAxisV); if(axisVLine) root.add(axisVLine);
  axisULine=makeAxisLine(aU,3,matAxisU); if(axisULine) root.add(axisULine);

  // Eigenvectors as singular-value-scaled V columns
  eigGroup=new THREE.Group();
  for(let i=0;i<3;i++){
    const scale=Math.abs(currentSVD.Sigma[i][i])*0.9;
    const d=currentSVD.V.map(r=>r[i]);
    const geo=new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(0,0,0),
      new THREE.Vector3(d[0]*scale,d[1]*scale,d[2]*scale),
    ]);
    eigGroup.add(new THREE.Line(geo,EIG_MATS[i]));
  }
  root.add(eigGroup);

  updateSceneForT();
  updateHUD();
}

function updateSceneForT() {
  const M=svdPath(tParam,currentSVD);
  const tPts=applyM(points,M);
  const tCube=applyM(cubeCorners,M);

  for(let i=0;i<points.length;i++){
    transGroup.children[i].position.set(...tPts[i]);
  }
  updateLineSegs(cubeTLine, cubePositions(tCube));
  updateLineSegs(dispLine,  dispPositions(points,tPts));

  // Axis V fades out during stage 2, axis U fades in at stage 3
  if(axisVLine){
    const op=tParam<=1?1:tParam<=2?Math.max(0,2-tParam):0;
    matAxisV.opacity=op; axisVLine.visible=op>0;
  }
  if(axisULine){
    const op=tParam>2?Math.min(1,tParam-2):0;
    matAxisU.opacity=op; axisULine.visible=op>0;
  }
}

// ─── VR Controllers ───────────────────────────────────────────────────────────

const ctrl1=renderer.xr.getController(0);
const ctrl2=renderer.xr.getController(1);
scene.add(ctrl1); scene.add(ctrl2);

function addRay(c){
  const g=new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(0,0,0),new THREE.Vector3(0,0,-1)]);
  const l=new THREE.Line(g,new THREE.LineBasicMaterial({color:0xffffff,transparent:true,opacity:0.5}));
  l.scale.z=3; c.add(l);
}
addRay(ctrl1); addRay(ctrl2);

let rHeld=false, lHeld=false;
ctrl1.addEventListener('selectstart',()=>rHeld=true);
ctrl1.addEventListener('selectend',  ()=>rHeld=false);
ctrl2.addEventListener('selectstart',()=>lHeld=true);
ctrl2.addEventListener('selectend',  ()=>lHeld=false);

ctrl1.addEventListener('squeezestart',()=>{ presetIdx=(presetIdx+1)%PRESETS.length; tParam=0; rebuildScene(); });

// ─── Keyboard ─────────────────────────────────────────────────────────────────

const keys={};
window.addEventListener('keydown',e=>{
  keys[e.key]=true;
  if(e.key==='1'){presetIdx=0;tParam=0;rebuildScene();}
  if(e.key==='2'){presetIdx=1;tParam=0;rebuildScene();}
  if(e.key==='3'){presetIdx=2;tParam=0;rebuildScene();}
  if(e.key.toLowerCase()==='g'){presetIdx=(presetIdx+1)%PRESETS.length;tParam=0;rebuildScene();}
});
window.addEventListener('keyup',e=>{ keys[e.key]=false; });

window.addEventListener('resize',()=>{
  camera.aspect=window.innerWidth/window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth,window.innerHeight);
});

// ─── Init ─────────────────────────────────────────────────────────────────────

rebuildScene();

const clock=new THREE.Clock();
const T_SPEED=0.7;

renderer.setAnimationLoop(()=>{
  const dt=Math.min(clock.getDelta(),0.05);
  let moved=false;

  if(rHeld||keys['ArrowRight']){ tParam=Math.min(3,tParam+T_SPEED*dt); moved=true; }
  if(lHeld||keys['ArrowLeft'] ){ tParam=Math.max(0,tParam-T_SPEED*dt); moved=true; }

  if(moved){ updateSceneForT(); updateHUD(); }

  renderer.render(scene,camera);
});
