import * as THREE from 'three';
import { VRButton } from 'three/examples/jsm/webxr/VRButton.js';
import { EffectComposer } from 'three/examples/jsm/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/examples/jsm/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/examples/jsm/postprocessing/UnrealBloomPass.js';
import { OutputPass } from 'three/examples/jsm/postprocessing/OutputPass.js';

// ─── 3×3 Math helpers ────────────────────────────────────────────────────────

function det3(m){return m[0][0]*(m[1][1]*m[2][2]-m[1][2]*m[2][1])-m[0][1]*(m[1][0]*m[2][2]-m[1][2]*m[2][0])+m[0][2]*(m[1][0]*m[2][1]-m[1][1]*m[2][0]);}
function mul(A,B){const C=[[0,0,0],[0,0,0],[0,0,0]];for(let i=0;i<3;i++)for(let j=0;j<3;j++)for(let k=0;k<3;k++)C[i][j]+=A[i][k]*B[k][j];return C;}
function mulVec(M,v){return[M[0][0]*v[0]+M[0][1]*v[1]+M[0][2]*v[2],M[1][0]*v[0]+M[1][1]*v[1]+M[1][2]*v[2],M[2][0]*v[0]+M[2][1]*v[1]+M[2][2]*v[2]];}
function tr3(A){return[[A[0][0],A[1][0],A[2][0]],[A[0][1],A[1][1],A[2][1]],[A[0][2],A[1][2],A[2][2]]];}

function rotAxisAngle(ax,ang){
  const n=Math.sqrt(ax[0]**2+ax[1]**2+ax[2]**2);
  if(n<1e-12||Math.abs(ang)<1e-12)return[[1,0,0],[0,1,0],[0,0,1]];
  const[x,y,z]=ax.map(a=>a/n),c=Math.cos(ang),s=Math.sin(ang),C=1-c;
  return[[c+x*x*C,x*y*C-z*s,x*z*C+y*s],[y*x*C+z*s,c+y*y*C,y*z*C-x*s],[z*x*C-y*s,z*y*C+x*s,c+z*z*C]];
}
function axisAngle(R){
  const tr=Math.max(-1,Math.min(3,R[0][0]+R[1][1]+R[2][2]));
  const theta=Math.acos(Math.max(-1,Math.min(1,(tr-1)/2)));
  if(Math.abs(theta)<1e-8)return{axis:[0,0,1],angle:0};
  const rx=R[2][1]-R[1][2],ry=R[0][2]-R[2][0],rz=R[1][0]-R[0][1],n=Math.sqrt(rx*rx+ry*ry+rz*rz);
  return{axis:n<1e-12?[1,0,0]:[rx/n,ry/n,rz/n],angle:theta};
}

function jacobiEig3(S){
  let A=S.map(r=>[...r]),V=[[1,0,0],[0,1,0],[0,0,1]];
  for(let it=0;it<60;it++){
    let p=0,q=1;
    for(let i=0;i<3;i++)for(let j=i+1;j<3;j++)if(Math.abs(A[i][j])>Math.abs(A[p][q])){p=i;q=j;}
    if(Math.abs(A[p][q])<1e-14)break;
    const th=0.5*Math.atan2(2*A[p][q],A[q][q]-A[p][p]),c=Math.cos(th),s=Math.sin(th);
    const nA=A.map(r=>[...r]);
    for(let i=0;i<3;i++){const ip=A[i][p]*c+A[i][q]*s,iq=-A[i][p]*s+A[i][q]*c;nA[i][p]=ip;nA[p][i]=ip;nA[i][q]=iq;nA[q][i]=iq;}
    nA[p][p]=A[p][p]*c*c+2*A[p][q]*c*s+A[q][q]*s*s;nA[q][q]=A[p][p]*s*s-2*A[p][q]*c*s+A[q][q]*c*c;nA[p][q]=0;nA[q][p]=0;A=nA;
    for(let i=0;i<3;i++){const vp=V[i][p]*c+V[i][q]*s,vq=-V[i][p]*s+V[i][q]*c;V[i][p]=vp;V[i][q]=vq;}
  }
  return{vals:[A[0][0],A[1][1],A[2][2]],vecs:V};
}

function svd3(A){
  const{vals,vecs}=jacobiEig3(mul(tr3(A),A));
  const idx=[0,1,2].sort((a,b)=>vals[b]-vals[a]);
  const sig=idx.map(i=>Math.sqrt(Math.max(0,vals[i])));
  const Vm=[[0,0,0],[0,0,0],[0,0,0]];for(let j=0;j<3;j++)for(let i=0;i<3;i++)Vm[i][j]=vecs[i][idx[j]];
  const Um=[[0,0,0],[0,0,0],[0,0,0]];
  for(let j=0;j<3;j++){const Av=mulVec(A,[Vm[0][j],Vm[1][j],Vm[2][j]]),s=sig[j];if(s>1e-10)for(let i=0;i<3;i++)Um[i][j]=Av[i]/s;else for(let i=0;i<3;i++)Um[i][j]=(i===j)?1:0;}
  return{U:Um,S:sig,V:Vm};
}

function makeRotationalSVD(A){
  const{U,S,V}=svd3(A);
  const Rf=[[1,0,0],[0,1,0],[0,0,-1]];
  const Sm=[[S[0],0,0],[0,S[1],0],[0,0,S[2]]];
  const dU=det3(U),dV=det3(V);
  if(dU<0&&dV<0)return{U:mul(U,Rf),Sigma:Sm,          V:mul(V,Rf)};
  if(dU<0)      return{U:mul(U,Rf),Sigma:mul(Rf,Sm),   V};
  if(dV<0)      return{U,          Sigma:mul(Sm,Rf),   V:mul(V,Rf)};
  return{U,Sigma:Sm,V};
}

function svdPath(t,{U,Sigma,V}){
  const{axis:aV,angle:tV}=axisAngle(V),{axis:aU,angle:tU}=axisAngle(U);
  const s1=Sigma[0][0],s2=Sigma[1][1],s3=Sigma[2][2];
  if(t<=1)return rotAxisAngle(aV,t*tV);
  if(t<=2){const a=t-1;return mul(rotAxisAngle(aV,tV),[[1+a*(s1-1),0,0],[0,1+a*(s2-1),0],[0,0,1+a*(s3-1)]]);}
  return mul(mul(rotAxisAngle(aV,tV),Sigma),rotAxisAngle(aU,-(t-2)*tU));
}

// ─── Presets ──────────────────────────────────────────────────────────────────

const PRESETS=[
  {name:'Symmetric',     A:[[1.0,0.2,0.0],[0.2,1.2,0.1],[0.0,0.1,0.8]]},
  {name:'Rotation+Scale',A:(()=>{const c=Math.cos(Math.PI/5),s=Math.sin(Math.PI/5);return mul([[c,-s,0],[s,c,0],[0,0,1]],[[1.8,0,0],[0,0.7,0],[0,0,1.2]]);})()},
  {name:'Shear+Scale',   A:[[1.5,0.8,0.2],[0.0,1.0,0.5],[0.1,0.0,0.6]]},
];

// ─── Point cloud & cube helpers ───────────────────────────────────────────────

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

function computePointColors(pts){
  let xn=Infinity,yn=Infinity,zn=Infinity,xx=-Infinity,yx=-Infinity,zx=-Infinity;
  for(const p of pts){if(p[0]<xn)xn=p[0];if(p[0]>xx)xx=p[0];if(p[1]<yn)yn=p[1];if(p[1]>yx)yx=p[1];if(p[2]<zn)zn=p[2];if(p[2]>zx)zx=p[2];}
  const xR=xx-xn||1,yR=yx-yn||1,zR=zx-zn||1;
  return pts.map(p=>new THREE.Color(0.15+(p[0]-xn)/xR*0.85,0.15+(p[1]-yn)/yR*0.85,0.15+(p[2]-zn)/zR*0.85));
}

function applyMToFloatArray(M,base,out){
  const n=base.length/3;
  for(let i=0;i<n;i++){
    const x=base[i*3],y=base[i*3+1],z=base[i*3+2];
    out[i*3]  =M[0][0]*x+M[0][1]*y+M[0][2]*z;
    out[i*3+1]=M[1][0]*x+M[1][1]*y+M[1][2]*z;
    out[i*3+2]=M[2][0]*x+M[2][1]*y+M[2][2]*z;
  }
}

const EDGES=[[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]];
const applyM=(pts,M)=>pts.map(p=>mulVec(M,p));
const cubePosArray=c=>EDGES.flatMap(([i,j])=>[...c[i],...c[j]]);
const dispPosArray=(o,t)=>o.flatMap((p,i)=>[...p,...t[i]]);
const CUBE_SOLID_IDX=[0,2,1,0,3,2,4,5,6,4,6,7,0,4,7,0,7,3,1,2,6,1,6,5,0,1,5,0,5,4,3,7,6,3,6,2];

// ─── Three.js core ────────────────────────────────────────────────────────────

const scene=new THREE.Scene();
scene.background=new THREE.Color(0x0d0d1a);
scene.fog=new THREE.FogExp2(0x0d0d1a,0.035);

const camera=new THREE.PerspectiveCamera(70,window.innerWidth/window.innerHeight,0.01,200);
camera.position.set(0,1.6,0);

const renderer=new THREE.WebGLRenderer({antialias:true});
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setSize(window.innerWidth,window.innerHeight);
renderer.xr.enabled=true;
renderer.toneMapping=THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure=1.0;
document.body.appendChild(renderer.domElement);
document.body.appendChild(VRButton.createButton(renderer));

// ─── Bloom (desktop only) ─────────────────────────────────────────────────────

const composer=new EffectComposer(renderer);
composer.addPass(new RenderPass(scene,camera));
const bloomPass=new UnrealBloomPass(new THREE.Vector2(window.innerWidth,window.innerHeight),0.35,0.4,0.75);
composer.addPass(bloomPass);
composer.addPass(new OutputPass());

// ─── Lighting ─────────────────────────────────────────────────────────────────

scene.add(new THREE.HemisphereLight(0xffffff,0x223344,1.1));
const dl=new THREE.DirectionalLight(0xffffff,0.6);dl.position.set(5,8,5);scene.add(dl);

// ─── Starfield + nebula ───────────────────────────────────────────────────────

(function makeStarfield(){
  function layer(n,r1,r2,sz,op,col){
    const pos=new Float32Array(n*3);
    for(let i=0;i<n;i++){
      const th=Math.random()*Math.PI*2,ph=Math.acos(2*Math.random()-1),r=r1+Math.random()*(r2-r1);
      pos[i*3]=r*Math.sin(ph)*Math.cos(th);pos[i*3+1]=r*Math.sin(ph)*Math.sin(th);pos[i*3+2]=r*Math.cos(ph);
    }
    const geo=new THREE.BufferGeometry();geo.setAttribute('position',new THREE.BufferAttribute(pos,3));
    scene.add(new THREE.Points(geo,new THREE.PointsMaterial({color:col,size:sz,sizeAttenuation:true,transparent:true,opacity:op})));
  }
  layer(2000,88,105,0.12,0.85,0xffffff);
  layer(500, 80,100,0.28,0.60,0xaabbff);
  layer(120, 75, 92,0.55,0.45,0xffffff);

  // Nebula band (additive)
  const n=600,pos=new Float32Array(n*3),cols=new Float32Array(n*3);
  const nc=[new THREE.Color(0x1133aa),new THREE.Color(0x991133),new THREE.Color(0x118855)];
  for(let i=0;i<n;i++){
    const th=Math.random()*Math.PI*2,r=58+Math.random()*45;
    const bandY=(Math.random()-0.5)*18;
    pos[i*3]=r*Math.cos(th);pos[i*3+1]=bandY;pos[i*3+2]=r*Math.sin(th);
    const c=nc[Math.floor(Math.random()*nc.length)];
    cols[i*3]=c.r;cols[i*3+1]=c.g;cols[i*3+2]=c.b;
  }
  const geo=new THREE.BufferGeometry();geo.setAttribute('position',new THREE.BufferAttribute(pos,3));geo.setAttribute('color',new THREE.BufferAttribute(cols,3));
  scene.add(new THREE.Points(geo,new THREE.PointsMaterial({size:1.1,sizeAttenuation:true,transparent:true,opacity:0.13,vertexColors:true,blending:THREE.AdditiveBlending,depthWrite:false})));
})();

// ─── Floor ────────────────────────────────────────────────────────────────────

const floorGrid=new THREE.GridHelper(20,20,0x334455,0x223344);floorGrid.position.y=0.001;scene.add(floorGrid);
const floorMesh=new THREE.Mesh(new THREE.PlaneGeometry(20,20),new THREE.MeshBasicMaterial({visible:false,side:THREE.DoubleSide}));
floorMesh.rotation.x=-Math.PI/2;scene.add(floorMesh);
const reticle=new THREE.Mesh(new THREE.RingGeometry(0.12,0.18,32),new THREE.MeshBasicMaterial({color:0x44ffaa,side:THREE.DoubleSide}));
reticle.rotation.x=-Math.PI/2;reticle.visible=false;scene.add(reticle);

// ─── Info panel ───────────────────────────────────────────────────────────────

const PANEL_W=1024,PANEL_H=920;
const panelCanvas=document.createElement('canvas');panelCanvas.width=PANEL_W;panelCanvas.height=PANEL_H;
const panelCtx=panelCanvas.getContext('2d');
const panelTex=new THREE.CanvasTexture(panelCanvas);
const panelMesh=new THREE.Mesh(new THREE.PlaneGeometry(1.8,1.62),new THREE.MeshBasicMaterial({map:panelTex,side:THREE.DoubleSide,transparent:true,depthWrite:false}));
panelMesh.position.set(2.4,1.6,-1.8);panelMesh.lookAt(0,1.6,0);scene.add(panelMesh);

const SV_COLORS=['#00ddff','#ff44cc','#ffee00'];
const SV_HEX   =[0x00ddff,  0xff44cc,  0xffee00];

function drawRoundRect(ctx,x,y,w,h,r){
  ctx.beginPath();ctx.moveTo(x+r,y);ctx.lineTo(x+w-r,y);ctx.quadraticCurveTo(x+w,y,x+w,y+r);
  ctx.lineTo(x+w,y+h-r);ctx.quadraticCurveTo(x+w,y+h,x+w-r,y+h);
  ctx.lineTo(x+r,y+h);ctx.quadraticCurveTo(x,y+h,x,y+h-r);
  ctx.lineTo(x,y+r);ctx.quadraticCurveTo(x,y,x+r,y);ctx.closePath();
}
function stageName(t){
  if(t<0.01) return'Identity (I)';
  if(t<=1.0) return'Stage 1 — V rotation';
  if(t<=2.0) return'Stage 2 — Σ scaling';
  return'Stage 3 — U rotation → A';
}
function divider(ctx,y,W,IND){
  ctx.strokeStyle='rgba(80,120,255,0.22)';ctx.lineWidth=1;
  ctx.beginPath();ctx.moveTo(IND,y);ctx.lineTo(W-IND,y);ctx.stroke();return y+14;
}
function swatch(ctx,x,y,color,w=22,h=16){ctx.fillStyle=color;ctx.fillRect(x,y-13,w,h);}

function updatePanel(){
  const ctx=panelCtx,W=PANEL_W,H=PANEL_H;
  ctx.clearRect(0,0,W,H);
  ctx.fillStyle='rgba(8,8,24,0.93)';drawRoundRect(ctx,0,0,W,H,28);ctx.fill();
  ctx.strokeStyle='rgba(80,120,255,0.45)';ctx.lineWidth=2;drawRoundRect(ctx,1,1,W-2,H-2,28);ctx.stroke();
  const preset=PRESETS[presetIdx];const A=preset.A;const svd=currentSVD;
  let y=46,IND=30;
  ctx.fillStyle='#7799ff';ctx.font='bold 28px monospace';ctx.fillText(`Matrix: ${preset.name}`,IND,y);y+=38;
  ctx.fillStyle='#bbccee';ctx.font='21px monospace';ctx.fillText('A =',IND,y);
  for(let r=0;r<3;r++){const row=A[r].map(v=>String(v.toFixed(2)).padStart(6));ctx.fillText(`[ ${row.join('  ')} ]`,IND+60,y+r*29);}
  y+=3*29+8;y=divider(ctx,y,W,IND);
  ctx.fillStyle='#ffdd55';ctx.font='bold 25px monospace';ctx.fillText(`t = ${tParam.toFixed(2)}`,IND,y);
  ctx.fillStyle='#aaaaaa';ctx.font='23px monospace';ctx.fillText(stageName(tParam),IND+148,y);y+=38;
  y=divider(ctx,y,W,IND);
  ctx.fillStyle='#88bbff';ctx.font='bold 23px monospace';ctx.fillText('SVD:  A = U · Σ · Vᵀ',IND,y);y+=34;
  if(svd){
    const sv=[svd.Sigma[0][0],svd.Sigma[1][1],svd.Sigma[2][2]];
    ctx.fillStyle='#cccccc';ctx.font='20px monospace';
    ctx.fillText(`σ = [ ${sv.map(s=>s.toFixed(3)).join(',  ')} ]`,IND,y);y+=32;
    const{axis:aV,angle:thV}=axisAngle(svd.V);
    ctx.fillStyle='#cc88ff';ctx.fillText(`V: ${(thV*180/Math.PI).toFixed(1)}°  axis=(${aV.map(x=>x.toFixed(2)).join(', ')})`,IND,y);y+=32;
    const{axis:aU,angle:thU}=axisAngle(svd.U);
    ctx.fillStyle='#44cccc';ctx.fillText(`U: ${(thU*180/Math.PI).toFixed(1)}°  axis=(${aU.map(x=>x.toFixed(2)).join(', ')})`,IND,y);y+=34;
  }
  y=divider(ctx,y,W,IND);
  ctx.fillStyle='#88bbff';ctx.font='bold 23px monospace';ctx.fillText('Legend',IND,y);y+=32;
  ctx.font='19px monospace';
  for(const[col,label] of[['#ff4444','X axis →'],['#44ff44','Y axis ↑'],['#4488ff','Z axis ·']]){
    swatch(ctx,IND,y,col);ctx.fillStyle='#cccccc';ctx.fillText(label,IND+30,y);y+=27;
  }
  y+=4;
  const sv2=currentSVD?[currentSVD.Sigma[0][0],currentSVD.Sigma[1][1],currentSVD.Sigma[2][2]]:[1,1,1];
  for(let i=0;i<3;i++){
    swatch(ctx,IND,y,SV_COLORS[i]);
    ctx.fillStyle=SV_COLORS[i];ctx.fillText(`v${i+1}`,IND+30,y);
    ctx.fillStyle='#cccccc';ctx.fillText(`right singular vec  σ${i+1}=${sv2[i].toFixed(3)}`,IND+72,y);y+=27;
  }
  y+=4;
  ctx.setLineDash([8,5]);
  ctx.strokeStyle='#aa55ff';ctx.lineWidth=3;ctx.beginPath();ctx.moveTo(IND,y-8);ctx.lineTo(IND+22,y-8);ctx.stroke();ctx.setLineDash([]);
  ctx.fillStyle='#cc88ff';ctx.font='19px monospace';ctx.fillText('V rotation axis (stage 1)',IND+30,y);y+=27;
  ctx.setLineDash([8,5]);
  ctx.strokeStyle='#00cccc';ctx.lineWidth=3;ctx.beginPath();ctx.moveTo(IND,y-8);ctx.lineTo(IND+22,y-8);ctx.stroke();ctx.setLineDash([]);
  ctx.fillStyle='#44cccc';ctx.fillText('U rotation axis (stage 3)',IND+30,y);y+=32;
  y=divider(ctx,y,W,IND);
  ctx.fillStyle='#555577';ctx.font='17px monospace';
  ctx.fillText('R-trigger: advance t        L-trigger: reverse t',IND,y);y+=26;
  ctx.fillText('R-grip: next matrix         L-stick: teleport',IND,y);y+=26;
  ctx.fillText('R-stick Y: zoom in/out',IND,y);
  panelTex.needsUpdate=true;
}

// ─── Wrist HUD ────────────────────────────────────────────────────────────────

const wristCanvas=document.createElement('canvas');wristCanvas.width=256;wristCanvas.height=128;
const wristCtx=wristCanvas.getContext('2d');
const wristTex=new THREE.CanvasTexture(wristCanvas);
const wristMesh=new THREE.Mesh(
  new THREE.PlaneGeometry(0.10,0.05),
  new THREE.MeshBasicMaterial({map:wristTex,side:THREE.DoubleSide,transparent:true,depthWrite:false})
);
wristMesh.position.set(0,0.05,-0.02);
wristMesh.rotation.x=-Math.PI*0.38;
let wristHUDAttached=false;

function updateWristHUD(){
  wristCtx.clearRect(0,0,256,128);
  wristCtx.fillStyle='rgba(10,10,30,0.88)';wristCtx.fillRect(0,0,256,128);
  wristCtx.strokeStyle='rgba(80,120,255,0.5)';wristCtx.lineWidth=2;wristCtx.strokeRect(1,1,254,126);
  wristCtx.fillStyle='#ffdd55';wristCtx.font='bold 26px monospace';
  wristCtx.fillText(`t = ${tParam.toFixed(2)}`,10,36);
  wristCtx.fillStyle='#aaaacc';wristCtx.font='19px monospace';
  wristCtx.fillText(PRESETS[presetIdx].name,10,66);
  wristCtx.fillStyle='#88bbff';
  wristCtx.fillText(stageName(tParam).slice(0,22),10,98);
  wristTex.needsUpdate=true;
}

// ─── Desktop HUD ──────────────────────────────────────────────────────────────

const hud=document.createElement('div');
Object.assign(hud.style,{position:'absolute',top:'12px',left:'12px',color:'#fff',
  fontFamily:'monospace',fontSize:'15px',background:'rgba(0,0,0,0.6)',
  padding:'10px 16px',borderRadius:'8px',lineHeight:'1.8',pointerEvents:'none'});
document.body.appendChild(hud);

function updateHUD(){
  hud.innerHTML=`Matrix: <b>${PRESETS[presetIdx].name}</b> [1/2/3 | G]<br>`+
    `t = <b>${tParam.toFixed(2)}</b> / 3.00 &nbsp; <b>${stageName(tParam)}</b><br>`+
    `Zoom: <b>${rootScale.toFixed(2)}x</b><br>`+
    `<span style="color:#aaa;font-size:12px">Hold ← → to scrub &nbsp;|&nbsp; VR: triggers=scrub, R-grip=matrix, L-stick=teleport, R-stick=zoom</span>`;
}

// ─── Root group ───────────────────────────────────────────────────────────────

const root=new THREE.Group();
root.scale.setScalar(0.55);
root.position.set(0,1.2,-1.8);
scene.add(root);
let rootScale=0.55;

const dummy=new THREE.Object3D();

// ─── Shared materials ─────────────────────────────────────────────────────────

const matCO=new THREE.LineBasicMaterial({color:0x778899});
const matCT=new THREE.LineBasicMaterial({color:0xff3333});
const matD =new THREE.LineBasicMaterial({color:0x5588aa,transparent:true,opacity:0.35});
const matAV=new THREE.LineDashedMaterial({color:0xaa44ff,dashSize:0.14,gapSize:0.07,transparent:true,opacity:1});
const matAU=new THREE.LineDashedMaterial({color:0x00cccc,dashSize:0.14,gapSize:0.07,transparent:true,opacity:0});

// ─── Grid ─────────────────────────────────────────────────────────────────────

const G_VALS=[-1.5,-0.75,0,0.75,1.5];
let gridBaseX,gridBaseY,gridBaseZ;
let gridMeshX,gridMeshY,gridMeshZ;
const gridTmpX=new Float32Array(G_VALS.length**2*6);
const gridTmpY=new Float32Array(G_VALS.length**2*6);
const gridTmpZ=new Float32Array(G_VALS.length**2*6);

function buildGridBase(){
  const L=G_VALS.length**2*6;
  gridBaseX=new Float32Array(L);gridBaseY=new Float32Array(L);gridBaseZ=new Float32Array(L);
  let ix=0,iy=0,iz=0;
  for(const y of G_VALS)for(const z of G_VALS){gridBaseX[ix++]=-1.5;gridBaseX[ix++]=y;gridBaseX[ix++]=z;gridBaseX[ix++]=1.5;gridBaseX[ix++]=y;gridBaseX[ix++]=z;}
  for(const x of G_VALS)for(const z of G_VALS){gridBaseY[iy++]=x;gridBaseY[iy++]=-1.5;gridBaseY[iy++]=z;gridBaseY[iy++]=x;gridBaseY[iy++]=1.5;gridBaseY[iy++]=z;}
  for(const x of G_VALS)for(const y of G_VALS){gridBaseZ[iz++]=x;gridBaseZ[iz++]=y;gridBaseZ[iz++]=-1.5;gridBaseZ[iz++]=x;gridBaseZ[iz++]=y;gridBaseZ[iz++]=1.5;}
}

// ─── Scene object helpers ─────────────────────────────────────────────────────

function setLineSegs(obj,pos){
  const attr=obj.geometry.attributes.position;
  if(attr&&attr.array.length===pos.length){attr.array.set(pos);attr.needsUpdate=true;}
  else{obj.geometry.setAttribute('position',new THREE.Float32BufferAttribute(pos,3));}
}
function makeSegs(pos,mat){
  const g=new THREE.BufferGeometry();g.setAttribute('position',new THREE.Float32BufferAttribute(pos,3));
  return new THREE.LineSegments(g,mat);
}
function makeAxisLine(ax,len,mat){
  const n=Math.sqrt(ax[0]**2+ax[1]**2+ax[2]**2);if(n<1e-12)return null;
  const a=ax.map(x=>x/n);
  const g=new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(-a[0]*len,-a[1]*len,-a[2]*len),new THREE.Vector3(a[0]*len,a[1]*len,a[2]*len)]);
  const l=new THREE.Line(g,mat);l.computeLineDistances();return l;
}
function makeLabel(text,colorStr){
  const c=document.createElement('canvas');c.width=256;c.height=64;
  const ctx=c.getContext('2d');
  ctx.font='bold 24px monospace';
  ctx.shadowColor=colorStr;ctx.shadowBlur=10;
  ctx.fillStyle=colorStr;ctx.textAlign='center';ctx.fillText(text,128,44);
  const tex=new THREE.CanvasTexture(c);
  const mat=new THREE.SpriteMaterial({map:tex,transparent:true,depthWrite:false});
  const sprite=new THREE.Sprite(mat);sprite.scale.set(0.44,0.11,1);return sprite;
}

// ─── Trail system ─────────────────────────────────────────────────────────────

const TRAIL_N=6;
let trailPts=[],trailBufs=[],trailAges=[],trailWriteIdx=0,lastTrailT=-99;

function initTrails(){
  trailPts.forEach(p=>root.remove(p));trailPts=[];trailBufs=[];trailAges=[];
  for(let i=0;i<TRAIL_N;i++){
    const buf=new Float32Array(30*3);
    const geo=new THREE.BufferGeometry();geo.setAttribute('position',new THREE.Float32BufferAttribute(buf.slice(),3));
    const mat=new THREE.PointsMaterial({size:0.035,transparent:true,opacity:0,color:0xaaccff,sizeAttenuation:true,depthWrite:false});
    const pts=new THREE.Points(geo,mat);pts.visible=false;
    root.add(pts);trailPts.push(pts);trailBufs.push(buf);trailAges.push(-1);
  }
  trailWriteIdx=0;lastTrailT=-99;
}

function pushTrail(tPts){
  if(Math.abs(tParam-lastTrailT)<0.05)return;
  lastTrailT=tParam;
  const buf=trailBufs[trailWriteIdx];
  const n=Math.min(tPts.length,30);
  for(let i=0;i<n;i++){buf[i*3]=tPts[i][0];buf[i*3+1]=tPts[i][1];buf[i*3+2]=tPts[i][2];}
  trailPts[trailWriteIdx].geometry.setAttribute('position',new THREE.Float32BufferAttribute(buf.slice(),3));
  trailPts[trailWriteIdx].visible=true;trailAges[trailWriteIdx]=0;
  trailWriteIdx=(trailWriteIdx+1)%TRAIL_N;
}

// ─── Stage pulse ──────────────────────────────────────────────────────────────

let pulseRing=null,pulseAge=-1,lastTFloor=0;

function triggerPulse(){
  if(pulseRing)root.remove(pulseRing);
  const geo=new THREE.TorusGeometry(0.4,0.015,8,48);
  const mat=new THREE.MeshBasicMaterial({color:0xffffff,transparent:true,opacity:1,depthWrite:false});
  pulseRing=new THREE.Mesh(geo,mat);pulseRing.rotation.x=Math.PI/2;
  root.add(pulseRing);pulseAge=0;
}

// ─── Background colors per stage ─────────────────────────────────────────────

const BG_COLS=[new THREE.Color(0x0d0d1a),new THREE.Color(0x110d1f),new THREE.Color(0x1a0d0d)];

// ─── State ────────────────────────────────────────────────────────────────────

let tParam=0,presetIdx=0,currentSVD=null,points=[],cubeCorners=[],pointColors=[];
let origIM,transIM,cubeOL,cubeTL,dispL,axisVL,axisUL,solidCubeMesh;
let svLabels=[];

// ─── Rebuild scene ────────────────────────────────────────────────────────────

function rebuildScene(){
  root.clear();svLabels=[];
  root.add(new THREE.AxesHelper(1.8));

  const A=PRESETS[presetIdx].A;
  currentSVD=makeRotationalSVD(A);
  points=genPoints(30,42);
  cubeCorners=buildCubeCorners(points);
  pointColors=computePointColors(points);

  // InstancedMesh for original (ghost) and transformed points
  const sGeo=new THREE.SphereGeometry(0.05,8,6);
  origIM=new THREE.InstancedMesh(sGeo,new THREE.MeshLambertMaterial({color:0xffffff,transparent:true,opacity:0.32}),points.length);
  transIM=new THREE.InstancedMesh(sGeo,new THREE.MeshLambertMaterial({color:0xffffff}),points.length);
  for(let i=0;i<points.length;i++){
    dummy.position.set(...points[i]);dummy.updateMatrix();
    origIM.setMatrixAt(i,dummy.matrix);transIM.setMatrixAt(i,dummy.matrix);
    origIM.setColorAt(i,pointColors[i].clone().multiplyScalar(0.5));
    transIM.setColorAt(i,pointColors[i]);
  }
  origIM.instanceMatrix.needsUpdate=true;origIM.instanceColor.needsUpdate=true;
  transIM.instanceMatrix.needsUpdate=true;transIM.instanceColor.needsUpdate=true;
  root.add(origIM);root.add(transIM);

  // Cube wireframes + displacement lines
  cubeOL=makeSegs(cubePosArray(cubeCorners),matCO);root.add(cubeOL);
  cubeTL=makeSegs(cubePosArray(cubeCorners),matCT);root.add(cubeTL);
  dispL =makeSegs(dispPosArray(points,points),matD);root.add(dispL);

  // Transparent solid cube
  const scPos=new Float32Array(8*3);
  for(let i=0;i<8;i++){scPos[i*3]=cubeCorners[i][0];scPos[i*3+1]=cubeCorners[i][1];scPos[i*3+2]=cubeCorners[i][2];}
  const scGeo=new THREE.BufferGeometry();
  scGeo.setAttribute('position',new THREE.BufferAttribute(scPos,3));
  scGeo.setIndex(CUBE_SOLID_IDX);
  solidCubeMesh=new THREE.Mesh(scGeo,new THREE.MeshBasicMaterial({color:0xff6633,transparent:true,opacity:0.10,side:THREE.DoubleSide,depthWrite:false}));
  root.add(solidCubeMesh);

  // Deforming 3D grid
  buildGridBase();
  gridMeshX=makeSegs(gridBaseX,new THREE.LineBasicMaterial({color:0x662222,transparent:true,opacity:0.35}));
  gridMeshY=makeSegs(gridBaseY,new THREE.LineBasicMaterial({color:0x226622,transparent:true,opacity:0.35}));
  gridMeshZ=makeSegs(gridBaseZ,new THREE.LineBasicMaterial({color:0x222288,transparent:true,opacity:0.35}));
  root.add(gridMeshX);root.add(gridMeshY);root.add(gridMeshZ);

  // Rotation axes
  const{axis:aV}=axisAngle(currentSVD.V),{axis:aU}=axisAngle(currentSVD.U);
  axisVL=makeAxisLine(aV,3,matAV);if(axisVL)root.add(axisVL);
  axisUL=makeAxisLine(aU,3,matAU);if(axisUL)root.add(axisUL);

  // Singular vector arrows + floating labels
  for(let i=0;i<3;i++){
    const len=Math.max(0.15,Math.abs(currentSVD.Sigma[i][i])*0.85);
    const d=currentSVD.V.map(r=>r[i]);
    const dir=new THREE.Vector3(d[0],d[1],d[2]).normalize();
    const hLen=Math.max(0.08,len*0.22);
    root.add(new THREE.ArrowHelper(dir,new THREE.Vector3(0,0,0),len,SV_HEX[i],hLen,hLen*0.55));
    const lbl=makeLabel(`v${i+1}  σ=${currentSVD.Sigma[i][i].toFixed(2)}`,SV_COLORS[i]);
    lbl.position.set(d[0]*len*1.35,d[1]*len*1.35,d[2]*len*1.35);
    root.add(lbl);svLabels.push(lbl);
  }

  // Reset pulse / trails
  if(pulseRing){root.remove(pulseRing);pulseRing=null;}pulseAge=-1;lastTFloor=0;
  initTrails();
  updateSceneForT();updatePanel();updateHUD();
}

// ─── Update scene for current t ───────────────────────────────────────────────

function updateSceneForT(){
  const M=svdPath(tParam,currentSVD);
  const tPts=applyM(points,M);
  const tCube=applyM(cubeCorners,M);

  // Transformed points (InstancedMesh)
  for(let i=0;i<points.length;i++){
    dummy.position.set(...tPts[i]);dummy.updateMatrix();transIM.setMatrixAt(i,dummy.matrix);
  }
  transIM.instanceMatrix.needsUpdate=true;

  // Wireframes + displacement
  setLineSegs(cubeTL,cubePosArray(tCube));
  setLineSegs(dispL, dispPosArray(points,tPts));

  // Solid cube face positions
  const scPos=solidCubeMesh.geometry.attributes.position;
  for(let i=0;i<8;i++){scPos.setXYZ(i,tCube[i][0],tCube[i][1],tCube[i][2]);}
  scPos.needsUpdate=true;

  // Deforming grid
  applyMToFloatArray(M,gridBaseX,gridTmpX);setLineSegs(gridMeshX,gridTmpX);
  applyMToFloatArray(M,gridBaseY,gridTmpY);setLineSegs(gridMeshY,gridTmpY);
  applyMToFloatArray(M,gridBaseZ,gridTmpZ);setLineSegs(gridMeshZ,gridTmpZ);

  // Rotation axis visibility
  if(axisVL){const op=tParam<=1?1:tParam<=2?Math.max(0,2-tParam):0;matAV.opacity=op;axisVL.visible=op>0;}
  if(axisUL){const op=tParam>2?Math.min(1,tParam-2):0;matAU.opacity=op;axisUL.visible=op>0;}

  // Background color lerp across stages
  const st=Math.min(2,Math.floor(tParam));
  const fr=Math.min(1,tParam-Math.floor(tParam));
  const bg=BG_COLS[st].clone().lerp(BG_COLS[Math.min(2,st+1)],fr);
  scene.background=bg;scene.fog.color.copy(bg);

  return tPts;
}

// ─── Controllers ─────────────────────────────────────────────────────────────

const ctrl1=renderer.xr.getController(0);
const ctrl2=renderer.xr.getController(1);
scene.add(ctrl1);scene.add(ctrl2);

function addRay(c,col=0xffffff){
  const g=new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(0,0,0),new THREE.Vector3(0,0,-1)]);
  const l=new THREE.Line(g,new THREE.LineBasicMaterial({color:col,transparent:true,opacity:0.5}));
  l.scale.z=5;c.add(l);
}
addRay(ctrl1,0xffffff);addRay(ctrl2,0x44ffaa);

let prevGripPressed=false;

// ─── Teleportation ────────────────────────────────────────────────────────────

let baseRefSpace=null,teleportTarget=null,prevThumbPressed=false;
renderer.xr.addEventListener('sessionstart',()=>{baseRefSpace=renderer.xr.getReferenceSpace();wristHUDAttached=false;});
renderer.xr.addEventListener('sessionend',()=>{baseRefSpace=null;wristHUDAttached=false;});

const teleportRay=new THREE.Raycaster();
const tmpMatrix=new THREE.Matrix4();

function doTeleport(pos){
  if(!baseRefSpace||typeof XRRigidTransform==='undefined')return;
  const t=new XRRigidTransform({x:-pos.x,y:0,z:-pos.z,w:1},{x:0,y:0,z:0,w:1});
  renderer.xr.setReferenceSpace(baseRefSpace.getOffsetReferenceSpace(t));
}

// ─── Keyboard ─────────────────────────────────────────────────────────────────

const keys={};
window.addEventListener('keydown',e=>{
  keys[e.key]=true;
  if(e.key==='1'){presetIdx=0;tParam=0;rebuildScene();}
  if(e.key==='2'){presetIdx=1;tParam=0;rebuildScene();}
  if(e.key==='3'){presetIdx=2;tParam=0;rebuildScene();}
  if(e.key.toLowerCase()==='g'){presetIdx=(presetIdx+1)%PRESETS.length;tParam=0;rebuildScene();}
  if(e.key==='='||e.key==='+'){rootScale=Math.min(2.0,rootScale+0.1);root.scale.setScalar(rootScale);updateHUD();}
  if(e.key==='-'){rootScale=Math.max(0.1,rootScale-0.1);root.scale.setScalar(rootScale);updateHUD();}
});
window.addEventListener('keyup',e=>{keys[e.key]=false;});
window.addEventListener('resize',()=>{
  camera.aspect=window.innerWidth/window.innerHeight;camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth,window.innerHeight);
  composer.setSize(window.innerWidth,window.innerHeight);
  bloomPass.setSize(window.innerWidth,window.innerHeight);
});

// ─── Init ─────────────────────────────────────────────────────────────────────

rebuildScene();updateHUD();updateWristHUD();

// ─── Render loop ──────────────────────────────────────────────────────────────

const clock=new THREE.Clock();
const T_SPEED=0.7;

renderer.setAnimationLoop(()=>{
  const dt=Math.min(clock.getDelta(),0.05);
  let moved=false;

  // Desktop t scrub
  if(keys['ArrowRight']){tParam=Math.min(3,tParam+T_SPEED*dt);moved=true;}
  if(keys['ArrowLeft']) {tParam=Math.max(0,tParam-T_SPEED*dt);moved=true;}

  // VR input
  if(renderer.xr.isPresenting){
    const session=renderer.xr.getSession();
    let leftCtrl=null;
    if(session){
      for(let i=0;i<session.inputSources.length;i++){
        const src=session.inputSources[i];
        if(!src.gamepad)continue;
        const ctrlGrp=i===0?ctrl1:ctrl2;
        const trigger=src.gamepad.buttons[0]?.pressed??false;
        const grip   =src.gamepad.buttons[1]?.pressed??false;
        const thumbBtn=src.gamepad.buttons[3]?.pressed??false;

        if(src.handedness==='right'){
          if(trigger){tParam=Math.min(3,tParam+T_SPEED*dt);moved=true;}
          if(grip&&!prevGripPressed){presetIdx=(presetIdx+1)%PRESETS.length;tParam=0;rebuildScene();}
          prevGripPressed=grip;
          // Right thumbstick Y → zoom (negative Y = stick up = zoom in)
          const stickY=src.gamepad.axes[1]??0;
          if(Math.abs(stickY)>0.15){
            rootScale=THREE.MathUtils.clamp(rootScale-stickY*0.9*dt,0.10,2.0);
            root.scale.setScalar(rootScale);moved=true;
          }
        }
        if(src.handedness==='left'){
          leftCtrl=ctrlGrp;
          if(trigger){tParam=Math.max(0,tParam-T_SPEED*dt);moved=true;}
          if(thumbBtn&&!prevThumbPressed&&teleportTarget)doTeleport(teleportTarget);
          prevThumbPressed=thumbBtn;
          // Attach wrist HUD to left controller once
          if(!wristHUDAttached){ctrlGrp.add(wristMesh);wristHUDAttached=true;}
        }
      }
    }

    // Teleport raycasting from left controller
    const raySource=leftCtrl??ctrl2;
    tmpMatrix.identity().extractRotation(raySource.matrixWorld);
    teleportRay.ray.origin.setFromMatrixPosition(raySource.matrixWorld);
    teleportRay.ray.direction.set(0,0,-1).applyMatrix4(tmpMatrix);
    const hits=teleportRay.intersectObject(floorMesh);
    if(hits.length&&hits[0].distance<12){
      reticle.position.copy(hits[0].point);reticle.position.y+=0.01;
      reticle.visible=true;teleportTarget=hits[0].point.clone();
    } else {reticle.visible=false;teleportTarget=null;}
  } else {reticle.visible=false;}

  // Handle move: update scene, trails, panel, HUD, pulse
  if(moved){
    const tPts=updateSceneForT();
    pushTrail(tPts);
    updatePanel();updateHUD();updateWristHUD();
    // Stage change pulse
    const newFloor=Math.min(2,Math.floor(tParam));
    if(newFloor!==lastTFloor&&tParam>0.02){triggerPulse();lastTFloor=newFloor;}
  }

  // Animate trails (fade out)
  for(let i=0;i<TRAIL_N;i++){
    if(trailAges[i]>=0){
      trailAges[i]+=dt;
      const op=Math.max(0,0.28-trailAges[i]*0.35);
      trailPts[i].material.opacity=op;
      if(op===0){trailPts[i].visible=false;trailAges[i]=-1;}
    }
  }

  // Animate stage pulse ring
  if(pulseRing&&pulseAge>=0){
    pulseAge+=dt;
    const s=1+pulseAge*10;
    pulseRing.scale.set(s,s,s);
    pulseRing.material.opacity=Math.max(0,1-pulseAge*2.8);
    if(pulseAge>0.75){root.remove(pulseRing);pulseRing=null;pulseAge=-1;}
  }

  // Render: bloom on desktop, direct in VR
  if(renderer.xr.isPresenting){
    renderer.render(scene,camera);
  } else {
    composer.render();
  }
});
