import * as THREE from 'three';
import { VRButton } from 'three/examples/jsm/webxr/VRButton.js';

// ─── 3×3 Math helpers ────────────────────────────────────────────────────────

function det3(m) {
  return m[0][0]*(m[1][1]*m[2][2]-m[1][2]*m[2][1])
        -m[0][1]*(m[1][0]*m[2][2]-m[1][2]*m[2][0])
        +m[0][2]*(m[1][0]*m[2][1]-m[1][1]*m[2][0]);
}
function mul(A,B){
  const C=[[0,0,0],[0,0,0],[0,0,0]];
  for(let i=0;i<3;i++) for(let j=0;j<3;j++) for(let k=0;k<3;k++) C[i][j]+=A[i][k]*B[k][j];
  return C;
}
function mulVec(M,v){
  return [M[0][0]*v[0]+M[0][1]*v[1]+M[0][2]*v[2],
          M[1][0]*v[0]+M[1][1]*v[1]+M[1][2]*v[2],
          M[2][0]*v[0]+M[2][1]*v[1]+M[2][2]*v[2]];
}
function tr3(A){return [[A[0][0],A[1][0],A[2][0]],[A[0][1],A[1][1],A[2][1]],[A[0][2],A[1][2],A[2][2]]];}

// ─── Rodrigues & axis-angle ───────────────────────────────────────────────────

function rotAxisAngle(ax,ang){
  const n=Math.sqrt(ax[0]**2+ax[1]**2+ax[2]**2);
  if(n<1e-12||Math.abs(ang)<1e-12) return [[1,0,0],[0,1,0],[0,0,1]];
  const[x,y,z]=ax.map(a=>a/n),c=Math.cos(ang),s=Math.sin(ang),C=1-c;
  return [[c+x*x*C,x*y*C-z*s,x*z*C+y*s],[y*x*C+z*s,c+y*y*C,y*z*C-x*s],[z*x*C-y*s,z*y*C+x*s,c+z*z*C]];
}
function axisAngle(R){
  const tr=Math.max(-1,Math.min(3,R[0][0]+R[1][1]+R[2][2]));
  const theta=Math.acos(Math.max(-1,Math.min(1,(tr-1)/2)));
  if(Math.abs(theta)<1e-8) return{axis:[0,0,1],angle:0};
  const rx=R[2][1]-R[1][2],ry=R[0][2]-R[2][0],rz=R[1][0]-R[0][1];
  const n=Math.sqrt(rx*rx+ry*ry+rz*rz);
  return{axis:n<1e-12?[1,0,0]:[rx/n,ry/n,rz/n],angle:theta};
}

// ─── Jacobi 3×3 SVD ───────────────────────────────────────────────────────────

function jacobiEig3(S){
  let A=S.map(r=>[...r]),V=[[1,0,0],[0,1,0],[0,0,1]];
  for(let it=0;it<60;it++){
    let p=0,q=1;
    for(let i=0;i<3;i++) for(let j=i+1;j<3;j++) if(Math.abs(A[i][j])>Math.abs(A[p][q])){p=i;q=j;}
    if(Math.abs(A[p][q])<1e-14) break;
    const th=0.5*Math.atan2(2*A[p][q],A[q][q]-A[p][p]),c=Math.cos(th),s=Math.sin(th);
    const nA=A.map(r=>[...r]);
    for(let i=0;i<3;i++){
      const ip=A[i][p]*c+A[i][q]*s,iq=-A[i][p]*s+A[i][q]*c;
      nA[i][p]=ip;nA[p][i]=ip;nA[i][q]=iq;nA[q][i]=iq;
    }
    nA[p][p]=A[p][p]*c*c+2*A[p][q]*c*s+A[q][q]*s*s;
    nA[q][q]=A[p][p]*s*s-2*A[p][q]*c*s+A[q][q]*c*c;
    nA[p][q]=0;nA[q][p]=0;A=nA;
    for(let i=0;i<3;i++){const vp=V[i][p]*c+V[i][q]*s,vq=-V[i][p]*s+V[i][q]*c;V[i][p]=vp;V[i][q]=vq;}
  }
  return{vals:[A[0][0],A[1][1],A[2][2]],vecs:V};
}

function svd3(A){
  const{vals,vecs}=jacobiEig3(mul(tr3(A),A));
  const idx=[0,1,2].sort((a,b)=>vals[b]-vals[a]);
  const sig=idx.map(i=>Math.sqrt(Math.max(0,vals[i])));
  const Vm=[[0,0,0],[0,0,0],[0,0,0]];
  for(let j=0;j<3;j++) for(let i=0;i<3;i++) Vm[i][j]=vecs[i][idx[j]];
  const Um=[[0,0,0],[0,0,0],[0,0,0]];
  for(let j=0;j<3;j++){
    const Av=mulVec(A,[Vm[0][j],Vm[1][j],Vm[2][j]]),s=sig[j];
    if(s>1e-10) for(let i=0;i<3;i++) Um[i][j]=Av[i]/s;
    else        for(let i=0;i<3;i++) Um[i][j]=(i===j)?1:0;
  }
  return{U:Um,S:sig,V:Vm};
}

function makeRotationalSVD(A){
  const{U,S,V}=svd3(A);
  const Rf=[[1,0,0],[0,1,0],[0,0,-1]];
  const Sm=[[S[0],0,0],[0,S[1],0],[0,0,S[2]]];
  const dU=det3(U),dV=det3(V);
  if(dU<0&&dV<0) return{U:mul(U,Rf),Sigma:Sm,          V:mul(V,Rf)};
  if(dU<0)       return{U:mul(U,Rf),Sigma:mul(Rf,Sm),   V};
  if(dV<0)       return{U,          Sigma:mul(Sm,Rf),   V:mul(V,Rf)};
  return{U,Sigma:Sm,V};
}

function svdPath(t,{U,Sigma,V}){
  const{axis:aV,angle:tV}=axisAngle(V),{axis:aU,angle:tU}=axisAngle(U);
  const s1=Sigma[0][0],s2=Sigma[1][1],s3=Sigma[2][2];
  if(t<=1) return rotAxisAngle(aV,t*tV);
  if(t<=2){const a=t-1;return mul(rotAxisAngle(aV,tV),[[1+a*(s1-1),0,0],[0,1+a*(s2-1),0],[0,0,1+a*(s3-1)]]);}
  return mul(mul(rotAxisAngle(aV,tV),Sigma),rotAxisAngle(aU,-(t-2)*tU));
}

// ─── Presets ──────────────────────────────────────────────────────────────────

const PRESETS=[
  {name:'Symmetric',     A:[[1.0,0.2,0.0],[0.2,1.2,0.1],[0.0,0.1,0.8]]},
  {name:'Rotation+Scale',A:(()=>{const c=Math.cos(Math.PI/5),s=Math.sin(Math.PI/5);return mul([[c,-s,0],[s,c,0],[0,0,1]],[[1.8,0,0],[0,0.7,0],[0,0,1.2]]);})()},
  {name:'Shear+Scale',   A:[[1.5,0.8,0.2],[0.0,1.0,0.5],[0.1,0.0,0.6]]},
];

// ─── Point cloud & cube ───────────────────────────────────────────────────────

function genPoints(n,seed){
  let s=seed>>>0;
  const rng=()=>{s=(s*1664525+1013904223)&0xffffffff;return(s>>>0)/0xffffffff;};
  const rn=()=>Math.sqrt(-2*Math.log(rng()||1e-10))*Math.cos(2*Math.PI*rng());
  const L11=Math.sqrt(1.2-0.16),L21=(0.3-0.4*0.2)/L11,L22=Math.sqrt(Math.max(0,0.8-0.04-L21**2));
  return Array.from({length:n},()=>{const z=[rn(),rn(),rn()];return[z[0],0.4*z[0]+L11*z[1],0.2*z[0]+L21*z[1]+L22*z[2]];});
}

function buildCubeCorners(pts){
  let xn=Infinity,yn=Infinity,zn=Infinity,xx=-Infinity,yx=-Infinity,zx=-Infinity;
  for(const p of pts){if(p[0]<xn)xn=p[0];if(p[0]>xx)xx=p[0];if(p[1]<yn)yn=p[1];if(p[1]>yx)yx=p[1];if(p[2]<zn)zn=p[2];if(p[2]>zx)zx=p[2];}
  const cx=(xn+xx)/2,cy=(yn+yx)/2,cz=(zn+zx)/2,h=0.5*Math.max(xx-xn,yx-yn,zx-zn)*1.3+1e-6;
  return[[cx-h,cy-h,cz-h],[cx+h,cy-h,cz-h],[cx+h,cy+h,cz-h],[cx-h,cy+h,cz-h],
         [cx-h,cy-h,cz+h],[cx+h,cy-h,cz+h],[cx+h,cy+h,cz+h],[cx-h,cy+h,cz+h]];
}

const EDGES=[[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]];
const applyM=(pts,M)=>pts.map(p=>mulVec(M,p));
const cubePosArray=c=>EDGES.flatMap(([i,j])=>[...c[i],...c[j]]);
const dispPosArray=(o,t)=>o.flatMap((p,i)=>[...p,...t[i]]);

// ─── Three.js core ────────────────────────────────────────────────────────────

const scene=new THREE.Scene();
scene.background=new THREE.Color(0x0d0d1a);
scene.fog=new THREE.FogExp2(0x0d0d1a,0.04);

const camera=new THREE.PerspectiveCamera(70,window.innerWidth/window.innerHeight,0.01,100);
camera.position.set(0,1.6,0);

const renderer=new THREE.WebGLRenderer({antialias:true});
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setSize(window.innerWidth,window.innerHeight);
renderer.xr.enabled=true;
document.body.appendChild(renderer.domElement);
document.body.appendChild(VRButton.createButton(renderer));

scene.add(new THREE.HemisphereLight(0xffffff,0x223344,1.1));
const dl=new THREE.DirectionalLight(0xffffff,0.6);dl.position.set(5,8,5);scene.add(dl);

// ─── Floor ────────────────────────────────────────────────────────────────────

const grid=new THREE.GridHelper(20,20,0x334455,0x223344);
grid.position.y=0.001;
scene.add(grid);

// Invisible floor plane for teleport raycasting
const floorMesh=new THREE.Mesh(
  new THREE.PlaneGeometry(20,20),
  new THREE.MeshBasicMaterial({visible:false,side:THREE.DoubleSide})
);
floorMesh.rotation.x=-Math.PI/2;
scene.add(floorMesh);

// Teleport reticle
const reticle=new THREE.Mesh(
  new THREE.RingGeometry(0.12,0.18,32),
  new THREE.MeshBasicMaterial({color:0x44ffaa,side:THREE.DoubleSide})
);
reticle.rotation.x=-Math.PI/2;
reticle.visible=false;
scene.add(reticle);

// ─── Info panel (canvas texture) ─────────────────────────────────────────────

const PANEL_W=1024,PANEL_H=700;
const panelCanvas=document.createElement('canvas');
panelCanvas.width=PANEL_W;panelCanvas.height=PANEL_H;
const panelCtx=panelCanvas.getContext('2d');
const panelTex=new THREE.CanvasTexture(panelCanvas);
const panelMesh=new THREE.Mesh(
  new THREE.PlaneGeometry(1.8,1.23),
  new THREE.MeshBasicMaterial({map:panelTex,side:THREE.DoubleSide,transparent:true,depthWrite:false})
);
panelMesh.position.set(2.4,1.55,-1.8);
panelMesh.lookAt(0,1.55,-1.8);
panelMesh.rotateY(Math.PI/2); // face inward (toward -X, i.e. toward user starting area)
scene.add(panelMesh);

function drawRoundRect(ctx,x,y,w,h,r){
  ctx.beginPath();
  ctx.moveTo(x+r,y);ctx.lineTo(x+w-r,y);ctx.quadraticCurveTo(x+w,y,x+w,y+r);
  ctx.lineTo(x+w,y+h-r);ctx.quadraticCurveTo(x+w,y+h,x+w-r,y+h);
  ctx.lineTo(x+r,y+h);ctx.quadraticCurveTo(x,y+h,x,y+h-r);
  ctx.lineTo(x,y+r);ctx.quadraticCurveTo(x,y,x+r,y);
  ctx.closePath();
}

function stageName(t){
  if(t<0.01)  return 'Identity (I)';
  if(t<=1.0)  return 'Stage 1 — V rotation';
  if(t<=2.0)  return 'Stage 2 — Σ scaling';
  return 'Stage 3 — U rotation → A';
}

function updatePanel(){
  const ctx=panelCtx,W=PANEL_W,H=PANEL_H;
  ctx.clearRect(0,0,W,H);

  // Background
  ctx.fillStyle='rgba(8,8,24,0.93)';
  drawRoundRect(ctx,0,0,W,H,28);ctx.fill();
  ctx.strokeStyle='rgba(80,120,255,0.45)';ctx.lineWidth=2;
  drawRoundRect(ctx,1,1,W-2,H-2,28);ctx.stroke();

  const preset=PRESETS[presetIdx];
  const A=preset.A;
  const svd=currentSVD;
  let y=48,LH=36,IND=30;

  // Title
  ctx.fillStyle='#7799ff';ctx.font='bold 30px monospace';
  ctx.fillText(`Matrix: ${preset.name}`,IND,y);y+=LH+6;

  // Matrix A
  ctx.fillStyle='#bbccee';ctx.font='22px monospace';
  ctx.fillText('A =',IND,y);
  for(let r=0;r<3;r++){
    const row=A[r].map(v=>String(v.toFixed(2)).padStart(6));
    ctx.fillText(`[ ${row.join('  ')} ]`,IND+64,y+(r*30));
  }
  y+=30*3+10;

  // Divider
  ctx.strokeStyle='rgba(80,120,255,0.25)';ctx.lineWidth=1;
  ctx.beginPath();ctx.moveTo(IND,y);ctx.lineTo(W-IND,y);ctx.stroke();y+=16;

  // t + stage
  ctx.fillStyle='#ffdd55';ctx.font='bold 26px monospace';
  ctx.fillText(`t = ${tParam.toFixed(2)}`,IND,y);
  ctx.fillStyle='#aaaaaa';ctx.font='24px monospace';
  ctx.fillText(stageName(tParam),IND+155,y);y+=LH+6;

  // Divider
  ctx.strokeStyle='rgba(80,120,255,0.25)';ctx.lineWidth=1;
  ctx.beginPath();ctx.moveTo(IND,y);ctx.lineTo(W-IND,y);ctx.stroke();y+=16;

  // SVD header
  ctx.fillStyle='#88bbff';ctx.font='bold 24px monospace';
  ctx.fillText('SVD:  A = U · Σ · Vᵀ',IND,y);y+=LH+2;

  // Singular values
  if(svd){
    const sv=[svd.Sigma[0][0],svd.Sigma[1][1],svd.Sigma[2][2]];
    ctx.fillStyle='#cccccc';ctx.font='21px monospace';
    ctx.fillText(`σ  =  [ ${sv.map(s=>s.toFixed(3)).join(',  ')} ]`,IND,y);y+=LH;

    // V axis-angle
    const{axis:aV,angle:thV}=axisAngle(svd.V);
    ctx.fillStyle='#cc88ff';ctx.font='21px monospace';
    ctx.fillText(`V: ${(thV*180/Math.PI).toFixed(1)}°  axis=(${aV.map(x=>x.toFixed(2)).join(', ')})`,IND,y);y+=LH;

    // U axis-angle
    const{axis:aU,angle:thU}=axisAngle(svd.U);
    ctx.fillStyle='#44cccc';
    ctx.fillText(`U: ${(thU*180/Math.PI).toFixed(1)}°  axis=(${aU.map(x=>x.toFixed(2)).join(', ')})`,IND,y);y+=LH+6;
  }

  // Divider
  ctx.strokeStyle='rgba(80,120,255,0.25)';ctx.lineWidth=1;
  ctx.beginPath();ctx.moveTo(IND,y);ctx.lineTo(W-IND,y);ctx.stroke();y+=14;

  // Controls
  ctx.fillStyle='#666688';ctx.font='18px monospace';
  ctx.fillText('R-trigger: advance t          L-trigger: reverse t',IND,y);y+=28;
  ctx.fillText('R-grip: next matrix           L-stick click: teleport',IND,y);

  panelTex.needsUpdate=true;
}

// ─── Visualization scene objects ──────────────────────────────────────────────

const root=new THREE.Group();
root.scale.setScalar(0.55);
root.position.set(0,1.2,-1.8);
scene.add(root);

const sphereGeo=new THREE.SphereGeometry(0.05,8,6);
const matO=new THREE.MeshLambertMaterial({color:0x4488ff});
const matT=new THREE.MeshLambertMaterial({color:0xff7722});
const matCO=new THREE.LineBasicMaterial({color:0x778899});
const matCT=new THREE.LineBasicMaterial({color:0xff3333});
const matD=new THREE.LineBasicMaterial({color:0x5588aa,transparent:true,opacity:0.35});
const matAV=new THREE.LineDashedMaterial({color:0xaa44ff,dashSize:0.14,gapSize:0.07,transparent:true,opacity:1});
const matAU=new THREE.LineDashedMaterial({color:0x00cccc,dashSize:0.14,gapSize:0.07,transparent:true,opacity:0});
const EIG_M=[new THREE.LineBasicMaterial({color:0x44ee66}),new THREE.LineBasicMaterial({color:0xffaa33}),new THREE.LineBasicMaterial({color:0xffee44})];

let origGrp,transGrp,cubeOL,cubeTL,dispL,axisVL,axisUL;

function setLineSegs(obj,pos){
  obj.geometry.setAttribute('position',new THREE.Float32BufferAttribute(pos,3));
  obj.geometry.attributes.position.needsUpdate=true;
}
function makeSegs(pos,mat){
  const g=new THREE.BufferGeometry();g.setAttribute('position',new THREE.Float32BufferAttribute(pos,3));
  return new THREE.LineSegments(g,mat);
}
function makeAxisLine(ax,len,mat){
  const n=Math.sqrt(ax[0]**2+ax[1]**2+ax[2]**2);
  if(n<1e-12)return null;
  const a=ax.map(x=>x/n);
  const g=new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(-a[0]*len,-a[1]*len,-a[2]*len),new THREE.Vector3(a[0]*len,a[1]*len,a[2]*len)]);
  const l=new THREE.Line(g,mat);l.computeLineDistances();return l;
}

// ─── State ────────────────────────────────────────────────────────────────────

let tParam=0,presetIdx=0,currentSVD=null,points=[],cubeCorners=[];

function rebuildScene(){
  root.clear();
  root.add(new THREE.AxesHelper(1.8));
  const A=PRESETS[presetIdx].A;
  currentSVD=makeRotationalSVD(A);
  points=genPoints(30,42);
  cubeCorners=buildCubeCorners(points);

  origGrp=new THREE.Group();
  transGrp=new THREE.Group();
  for(const p of points){
    const mo=new THREE.Mesh(sphereGeo,matO);mo.position.set(...p);origGrp.add(mo);
    const mt=new THREE.Mesh(sphereGeo,matT);mt.position.set(...p);transGrp.add(mt);
  }
  root.add(origGrp);root.add(transGrp);

  cubeOL=makeSegs(cubePosArray(cubeCorners),matCO);root.add(cubeOL);
  cubeTL=makeSegs(cubePosArray(cubeCorners),matCT);root.add(cubeTL);
  dispL =makeSegs(dispPosArray(points,points),matD);root.add(dispL);

  const{axis:aV}=axisAngle(currentSVD.V),{axis:aU}=axisAngle(currentSVD.U);
  axisVL=makeAxisLine(aV,3,matAV);if(axisVL)root.add(axisVL);
  axisUL=makeAxisLine(aU,3,matAU);if(axisUL)root.add(axisUL);

  // Singular-vector lines (V columns scaled by σ)
  for(let i=0;i<3;i++){
    const sc=Math.abs(currentSVD.Sigma[i][i])*0.9;
    const d=currentSVD.V.map(r=>r[i]);
    const g=new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(0,0,0),new THREE.Vector3(d[0]*sc,d[1]*sc,d[2]*sc)]);
    root.add(new THREE.Line(g,EIG_M[i]));
  }

  updateSceneForT();
  updatePanel();
}

function updateSceneForT(){
  const M=svdPath(tParam,currentSVD);
  const tPts=applyM(points,M),tCube=applyM(cubeCorners,M);
  for(let i=0;i<points.length;i++) transGrp.children[i].position.set(...tPts[i]);
  setLineSegs(cubeTL,cubePosArray(tCube));
  setLineSegs(dispL, dispPosArray(points,tPts));
  if(axisVL){const op=tParam<=1?1:tParam<=2?Math.max(0,2-tParam):0;matAV.opacity=op;axisVL.visible=op>0;}
  if(axisUL){const op=tParam>2?Math.min(1,tParam-2):0;matAU.opacity=op;axisUL.visible=op>0;}
}

// ─── Controllers ─────────────────────────────────────────────────────────────

const ctrl1=renderer.xr.getController(0); // right
const ctrl2=renderer.xr.getController(1); // left
scene.add(ctrl1);scene.add(ctrl2);

function addRay(c,col=0xffffff){
  const g=new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(0,0,0),new THREE.Vector3(0,0,-1)]);
  const l=new THREE.Line(g,new THREE.LineBasicMaterial({color:col,transparent:true,opacity:0.5}));
  l.scale.z=5;c.add(l);
}
addRay(ctrl1,0xffffff);
addRay(ctrl2,0x44ffaa); // green ray on left = teleport pointer

let rHeld=false,lHeld=false;
ctrl1.addEventListener('selectstart',()=>rHeld=true);
ctrl1.addEventListener('selectend',  ()=>rHeld=false);
ctrl2.addEventListener('selectstart',()=>lHeld=true);
ctrl2.addEventListener('selectend',  ()=>lHeld=false);
ctrl1.addEventListener('squeezestart',()=>{presetIdx=(presetIdx+1)%PRESETS.length;tParam=0;rebuildScene();});

// ─── Teleportation ────────────────────────────────────────────────────────────

let baseRefSpace=null;
let teleportTarget=null;
let prevThumbPressed=false;

renderer.xr.addEventListener('sessionstart',()=>{
  baseRefSpace=renderer.xr.getReferenceSpace();
});

const teleportRay=new THREE.Raycaster();
const tmpMatrix=new THREE.Matrix4();

function doTeleport(pos){
  if(!baseRefSpace||typeof XRRigidTransform==='undefined') return;
  const t=new XRRigidTransform({x:-pos.x,y:0,z:-pos.z,w:1},{x:0,y:0,z:0,w:1});
  renderer.xr.setReferenceSpace(baseRefSpace.getOffsetReferenceSpace(t));
}

// ─── Desktop HUD ──────────────────────────────────────────────────────────────

const hud=document.createElement('div');
Object.assign(hud.style,{position:'absolute',top:'12px',left:'12px',color:'#fff',
  fontFamily:'monospace',fontSize:'15px',background:'rgba(0,0,0,0.6)',
  padding:'10px 16px',borderRadius:'8px',lineHeight:'1.8',pointerEvents:'none'});
document.body.appendChild(hud);

function updateHUD(){
  hud.innerHTML=
    `Matrix: <b>${PRESETS[presetIdx].name}</b> [1/2/3 | G]<br>`+
    `t = <b>${tParam.toFixed(2)}</b> / 3.00 &nbsp; <b>${stageName(tParam)}</b><br>`+
    `<span style="color:#aaa;font-size:12px">Hold ← → to scrub &nbsp;|&nbsp; In VR: triggers scrub, grip=matrix, L-stick=teleport</span>`;
}

// ─── Keyboard ─────────────────────────────────────────────────────────────────

const keys={};
window.addEventListener('keydown',e=>{
  keys[e.key]=true;
  if(e.key==='1'){presetIdx=0;tParam=0;rebuildScene();}
  if(e.key==='2'){presetIdx=1;tParam=0;rebuildScene();}
  if(e.key==='3'){presetIdx=2;tParam=0;rebuildScene();}
  if(e.key.toLowerCase()==='g'){presetIdx=(presetIdx+1)%PRESETS.length;tParam=0;rebuildScene();}
});
window.addEventListener('keyup',e=>{keys[e.key]=false;});
window.addEventListener('resize',()=>{
  camera.aspect=window.innerWidth/window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth,window.innerHeight);
});

// ─── Init & render loop ───────────────────────────────────────────────────────

rebuildScene();

const clock=new THREE.Clock();
const T_SPEED=0.7;

renderer.setAnimationLoop(()=>{
  const dt=Math.min(clock.getDelta(),0.05);
  let moved=false;

  // Scrub t
  if(rHeld||keys['ArrowRight']){tParam=Math.min(3,tParam+T_SPEED*dt);moved=true;}
  if(lHeld||keys['ArrowLeft']) {tParam=Math.max(0,tParam-T_SPEED*dt);moved=true;}
  if(moved){updateSceneForT();updateHUD();updatePanel();}

  // Teleport raycasting (VR only)
  if(renderer.xr.isPresenting){
    tmpMatrix.identity().extractRotation(ctrl2.matrixWorld);
    teleportRay.ray.origin.setFromMatrixPosition(ctrl2.matrixWorld);
    teleportRay.ray.direction.set(0,0,-1).applyMatrix4(tmpMatrix);
    const hits=teleportRay.intersectObject(floorMesh);
    if(hits.length&&hits[0].distance<12){
      reticle.position.copy(hits[0].point);
      reticle.position.y+=0.01;
      reticle.visible=true;
      teleportTarget=hits[0].point.clone();
    } else {
      reticle.visible=false;
      teleportTarget=null;
    }

    // Thumbstick click (button 3) on left controller
    const session=renderer.xr.getSession();
    if(session){
      for(const src of session.inputSources){
        if(src.handedness==='left'&&src.gamepad){
          const pressed=src.gamepad.buttons[3]?.pressed??false;
          if(pressed&&!prevThumbPressed&&teleportTarget) doTeleport(teleportTarget);
          prevThumbPressed=pressed;
        }
      }
    }
  } else {
    reticle.visible=false;
  }

  renderer.render(scene,camera);
});
