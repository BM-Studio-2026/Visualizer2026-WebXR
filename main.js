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

// ─── SVD helpers for non-square matrices ─────────────────────────────────────

function svd2x3(A){
  // A: 2×3 → {U:2×2, s:[s1,s2], V:3×3 columns=right sing vecs}
  const AtA=[[0,0,0],[0,0,0],[0,0,0]];
  for(let i=0;i<3;i++)for(let j=0;j<3;j++)AtA[i][j]=A[0][i]*A[0][j]+A[1][i]*A[1][j];
  const{vals,vecs}=jacobiEig3(AtA);
  const idx=[0,1,2].sort((a,b)=>vals[b]-vals[a]);
  const sig=idx.map(i=>Math.sqrt(Math.max(0,vals[i])));
  const V=[[0,0,0],[0,0,0],[0,0,0]];
  for(let j=0;j<3;j++)for(let i=0;i<3;i++)V[i][j]=vecs[i][idx[j]];
  if(det3(V)<0)for(let i=0;i<3;i++)V[i][2]*=-1;
  const U=[[0,0],[0,0]];
  for(let j=0;j<2;j++){
    if(sig[j]>1e-10){
      const u=[A[0][0]*V[0][j]+A[0][1]*V[1][j]+A[0][2]*V[2][j],
               A[1][0]*V[0][j]+A[1][1]*V[1][j]+A[1][2]*V[2][j]];
      U[0][j]=u[0]/sig[j];U[1][j]=u[1]/sig[j];
    }else{U[0][j]=(j===0?1:0);U[1][j]=(j===0?0:1);}
  }
  return{U,s:sig.slice(0,2),V};
}

function svd3x2(A){
  // A: 3×2 → {U:3×3, s:[s1,s2], V:2×2}
  const AAt=[[0,0,0],[0,0,0],[0,0,0]];
  for(let i=0;i<3;i++)for(let j=0;j<3;j++)AAt[i][j]=A[i][0]*A[j][0]+A[i][1]*A[j][1];
  const{vals,vecs}=jacobiEig3(AAt);
  const idx=[0,1,2].sort((a,b)=>vals[b]-vals[a]);
  const sig=idx.map(i=>Math.sqrt(Math.max(0,vals[i])));
  const U=[[0,0,0],[0,0,0],[0,0,0]];
  for(let j=0;j<3;j++)for(let i=0;i<3;i++)U[i][j]=vecs[i][idx[j]];
  if(det3(U)<0)for(let i=0;i<3;i++)U[i][2]*=-1;
  const V=[[0,0],[0,0]];
  for(let j=0;j<2;j++){
    if(sig[j]>1e-10){
      const v=[A[0][0]*U[0][j]+A[1][0]*U[1][j]+A[2][0]*U[2][j],
               A[0][1]*U[0][j]+A[1][1]*U[1][j]+A[2][1]*U[2][j]];
      V[0][j]=v[0]/sig[j];V[1][j]=v[1]/sig[j];
    }else{V[0][j]=(j===0?1:0);V[1][j]=(j===0?0:1);}
  }
  return{U,s:sig.slice(0,2),V};
}

function genPoints2D(n,seed){
  let s=seed>>>0;
  const rng=()=>{s=(s*1664525+1013904223)&0xffffffff;return(s>>>0)/0xffffffff;};
  const rn=()=>Math.sqrt(-2*Math.log(rng()||1e-10))*Math.cos(2*Math.PI*rng());
  const L=Math.sqrt(1.2-0.16);
  return Array.from({length:n},()=>{const a=rn(),b=rn();return[a,0.4*a+L*b];});
}

function genPancake3D(n,seed){
  let s=seed>>>0;
  const rng=()=>{s=(s*1664525+1013904223)&0xffffffff;return(s>>>0)/0xffffffff;};
  const rn=()=>Math.sqrt(-2*Math.log(rng()||1e-10))*Math.cos(2*Math.PI*rng());
  const ax=25*Math.PI/180,ay=35*Math.PI/180;
  const Rx=[[1,0,0],[0,Math.cos(ax),-Math.sin(ax)],[0,Math.sin(ax),Math.cos(ax)]];
  const Ry=[[Math.cos(ay),0,Math.sin(ay)],[0,1,0],[-Math.sin(ay),0,Math.cos(ay)]];
  const R=mul(Ry,Rx);
  return Array.from({length:n},()=>mulVec(R,[rn()*2.5,rn()*1.4,rn()*0.3]));
}

function pca3(pts){
  const n=pts.length,mean=[0,0,0];
  for(const p of pts){mean[0]+=p[0];mean[1]+=p[1];mean[2]+=p[2];}
  mean[0]/=n;mean[1]/=n;mean[2]/=n;
  const centered=pts.map(p=>[p[0]-mean[0],p[1]-mean[1],p[2]-mean[2]]);
  const C=[[0,0,0],[0,0,0],[0,0,0]];
  for(const p of centered)for(let i=0;i<3;i++)for(let j=0;j<3;j++)C[i][j]+=p[i]*p[j];
  const{vals,vecs}=jacobiEig3(C);
  const idx=[0,1,2].sort((a,b)=>vals[b]-vals[a]);
  const sig=idx.map(i=>Math.sqrt(Math.max(0,vals[i]/n)));
  const evals=idx.map(i=>Math.max(0,vals[i]/n));
  const Cov=C.map(row=>row.map(v=>v/n));
  const V=[[0,0,0],[0,0,0],[0,0,0]];
  for(let j=0;j<3;j++)for(let i=0;i<3;i++)V[i][j]=vecs[i][idx[j]];
  if(det3(V)<0)for(let i=0;i<3;i++)V[i][2]*=-1;
  return{mean,V,s:sig,centered,Cov,evals};
}

function lseSolve(planes){
  const An=[],b=[];
  for(const[a,bv,c,d] of planes){
    const nm=Math.sqrt(a*a+bv*bv+c*c);if(nm<1e-10)continue;
    An.push([a/nm,bv/nm,c/nm]);b.push(-d/nm);
  }
  const AtA=[[0,0,0],[0,0,0],[0,0,0]],Atb=[0,0,0];
  for(let i=0;i<An.length;i++){
    for(let r=0;r<3;r++)for(let c2=0;c2<3;c2++)AtA[r][c2]+=An[i][r]*An[i][c2];
    for(let r=0;r<3;r++)Atb[r]+=An[i][r]*b[i];
  }
  const d=det3(AtA);
  let xLS=[0,0,0];
  if(Math.abs(d)>1e-12){
    const col=(M,ci,v)=>M.map((row,ri)=>row.map((x,j)=>j===ci?v[ri]:x));
    xLS=[det3(col(AtA,0,Atb)),det3(col(AtA,1,Atb)),det3(col(AtA,2,Atb))].map(v=>v/d);
  }
  const dists=An.map((n,i)=>Math.abs(n[0]*xLS[0]+n[1]*xLS[1]+n[2]*xLS[2]-b[i]));
  return{xLS,normals:An,b,dists};
}

// ─── Animation paths for new scenarios ───────────────────────────────────────

function pathScen1(pts,t,svd){
  // 2×3: R³ → R²  (points collapse onto V₁,V₂ plane)
  const{U,s,V}=svd,{axis:aV,angle:tV}=axisAngle(V);
  if(t<=1){const Rt=rotAxisAngle(aV,t*tV);return pts.map(p=>mulVec(Rt,p));}
  const Vt=tr3(V),pVF=pts.map(p=>mulVec(Vt,p));
  if(t<=2){
    const b=t-1,s1t=1+b*(s[0]-1),s2t=1+b*(s[1]-1),zt=1-b;
    return pVF.map(p=>mulVec(V,[p[0]*s1t,p[1]*s2t,p[2]*zt]));
  }
  const g=t-2,tU=Math.atan2(U[1][0],U[0][0]),cU=Math.cos(g*tU),sU=Math.sin(g*tU);
  return pVF.map(p=>{const x=p[0]*s[0],y=p[1]*s[1];return mulVec(V,[cU*x-sU*y,sU*x+cU*y,0]);});
}

function pathScen2(pts2d,t,svd){
  // 3×2: R² → R³  (points lift from z=0 plane into 3D)
  const{U,s,V}=svd,tV=Math.atan2(V[1][0],V[0][0]);
  if(t<=1){
    const cV=Math.cos(t*tV),sV=Math.sin(t*tV);
    return pts2d.map(p=>[cV*p[0]-sV*p[1],sV*p[0]+cV*p[1],0]);
  }
  const pR=pts2d.map(p=>[V[0][0]*p[0]+V[0][1]*p[1],V[1][0]*p[0]+V[1][1]*p[1]]);
  if(t<=2){
    const b=t-1,s1t=1+b*(s[0]-1),s2t=1+b*(s[1]-1);
    return pR.map(p=>[p[0]*s1t,p[1]*s2t,0]);
  }
  const{axis:aU,angle:tU}=axisAngle(U),R3=rotAxisAngle(aU,(t-2)*tU);
  return pR.map(p=>mulVec(R3,[p[0]*s[0],p[1]*s[1],0]));
}

function pathScen3(centered,t,pca){
  // PCA: rotate to PC frame, then squash PC3, then PC2
  const{V,s}=pca,Vt=tr3(V),{axis:aV,angle:tV}=axisAngle(V);
  if(t<=1){const Rt=rotAxisAngle(aV,t*tV);return centered.map(p=>mulVec(Rt,p));}
  const pPCF=centered.map(p=>mulVec(Vt,p));
  if(t<=2){const b=t-1;return pPCF.map(p=>mulVec(V,[p[0],p[1],p[2]*(1-b)]));}
  const g=t-2;return pPCF.map(p=>mulVec(V,[p[0],p[1]*(1-g),0]));
}

// ─── Bounding square helpers (for 2D input scenarios) ────────────────────────

function buildBoundingSquare2D(pts2d){
  let xn=Infinity,yn=Infinity,xx=-Infinity,yx=-Infinity;
  for(const p of pts2d){if(p[0]<xn)xn=p[0];if(p[0]>xx)xx=p[0];if(p[1]<yn)yn=p[1];if(p[1]>yx)yx=p[1];}
  const cx=(xn+xx)/2,cy=(yn+yx)/2,h=0.5*Math.max(xx-xn,yx-yn)*1.3+1e-6;
  return[[cx-h,cy-h],[cx+h,cy-h],[cx+h,cy+h],[cx-h,cy+h]];
}
const SQUARE_EDGES=[[0,1],[1,2],[2,3],[3,0]];
function squarePosFlat(corners){
  return SQUARE_EDGES.flatMap(([i,j])=>[corners[i][0],corners[i][1],0,corners[j][0],corners[j][1],0]);
}
function squarePos3D(corners3){
  return SQUARE_EDGES.flatMap(([i,j])=>[...corners3[i],...corners3[j]]);
}

// ─── Presets ──────────────────────────────────────────────────────────────────

const PRESETS=[
  {name:'Symmetric',     A:[[1.0,0.2,0.0],[0.2,1.2,0.1],[0.0,0.1,0.8]]},
  {name:'Rotation+Scale',A:(()=>{const c=Math.cos(Math.PI/5),s=Math.sin(Math.PI/5);return mul([[c,-s,0],[s,c,0],[0,0,1]],[[1.8,0,0],[0,0.7,0],[0,0,1.2]]);})()},
  {name:'Shear+Scale',   A:[[1.5,0.8,0.2],[0.0,1.0,0.5],[0.1,0.0,0.6]]},
];
function deepCopy2D(m){return m.map(r=>[...r]);}
const MAT1_DEFAULT=[[1,2,0],[0,1,-1]];
const MAT2_DEFAULT=[[1,2],[0,1],[-1,0]];
let mat0Custom=deepCopy2D(PRESETS[0].A);
let mat1Custom=deepCopy2D(MAT1_DEFAULT);
let mat2Custom=deepCopy2D(MAT2_DEFAULT);

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

// ─── Three.js core ────────────────────────────────────────────────────────────

const scene=new THREE.Scene();
scene.background=new THREE.Color(0x0d0d1a);
// no fog — deep space environment

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

// ─── Starfield + nebula (fallback; replaced by photo when loaded) ─────────────

const starfieldGrp=new THREE.Group();scene.add(starfieldGrp);
let hasSkyPhoto=false;

(function makeStarfield(){
  function pts(pos,col,size,opacity,blending=THREE.NormalBlending){
    const geo=new THREE.BufferGeometry();
    geo.setAttribute('position',new THREE.BufferAttribute(new Float32Array(pos),3));
    if(col) geo.setAttribute('color',new THREE.BufferAttribute(new Float32Array(col),3));
    starfieldGrp.add(new THREE.Points(geo,new THREE.PointsMaterial({
      size,sizeAttenuation:true,transparent:true,opacity,depthWrite:false,
      vertexColors:!!col,color:col?0xffffff:0xffffff,blending
    })));
  }

  // ── 1. Background field (random sphere) ─────────────────────
  {const n=3000,p=[];
   for(let i=0;i<n;i++){const th=Math.random()*Math.PI*2,ph=Math.acos(2*Math.random()-1),r=88+Math.random()*22;p.push(r*Math.sin(ph)*Math.cos(th),r*Math.sin(ph)*Math.sin(th),r*Math.cos(ph));}
   pts(p,null,0.10,0.80);}

  // ── 2. Colour-varied bright stars ───────────────────────────
  {const n=500,p=[],c=[];
   const cc=[[1,.95,.8],[.75,.85,1],[1,1,1],[1,.7,.45],[.6,.7,1],[.9,.6,.4]];
   for(let i=0;i<n;i++){
     const th=Math.random()*Math.PI*2,ph=Math.acos(2*Math.random()-1),r=84+Math.random()*18;
     p.push(r*Math.sin(ph)*Math.cos(th),r*Math.sin(ph)*Math.sin(th),r*Math.cos(ph));
     const col=cc[Math.floor(Math.random()*cc.length)];c.push(...col);
   }
   pts(p,c,0.38,0.80);}

  // ── 3. Milky Way band ────────────────────────────────────────
  // Galactic plane: normal = (nx,ny,nz), two in-plane axes (ux,uy,uz) and (vx,vy,vz)
  const gn=[0.18,0.92,0.35]; const gnL=Math.sqrt(gn[0]**2+gn[1]**2+gn[2]**2);
  const[nx,ny,nz]=gn.map(x=>x/gnL);
  // u = cross([0,1,0], n) if not parallel, else cross([1,0,0], n)
  const tmp=Math.abs(ny)<0.85?[0,1,0]:[1,0,0];
  const ux=tmp[1]*nz-tmp[2]*ny, uy=tmp[2]*nx-tmp[0]*nz, uz=tmp[0]*ny-tmp[1]*nx;
  const uL=Math.sqrt(ux**2+uy**2+uz**2);
  const[Ux,Uy,Uz]=[ux/uL,uy/uL,uz/uL];
  // v = cross(n, u)
  const Vx=ny*Uz-nz*Uy, Vy=nz*Ux-nx*Uz, Vz=nx*Uy-ny*Ux;

  {const n=16000,p=[],c=[];
   for(let i=0;i<n;i++){
     const a=Math.random()*Math.PI*2;
     // Gaussian scatter from band plane (σ≈0.055 rad ≈ 3°)
     const sc=(Math.random()+Math.random()+Math.random()-1.5)*0.11;
     const cosA=Math.cos(a),sinA=Math.sin(a),cosS=Math.cos(sc),sinS=Math.sin(sc)*( Math.random()<.5?1:-1);
     const ix=cosA*Ux+sinA*Vx, iy=cosA*Uy+sinA*Vy, iz=cosA*Uz+sinA*Vz;
     const r=83+Math.random()*24;
     p.push(r*(cosS*ix+sinS*nx), r*(cosS*iy+sinS*ny), r*(cosS*iz+sinS*nz));
     // Colour: warm toward galactic centre (a≈0), cool toward anti-centre (a≈π)
     const dCenter=Math.abs(Math.cos(a)); // 1 = centre, 0 = anti-centre
     const bright=0.35+Math.random()*0.45;
     if(dCenter>0.7){c.push(bright,bright*0.88,bright*0.55);} // warm gold
     else            {c.push(bright*0.7,bright*0.78,bright);} // cool blue
   }
   pts(p,c,0.075,0.50);}

  // ── 4. Galactic centre glow (extra dense warm cluster) ──────
  {const n=4000,p=[],c=[];
   for(let i=0;i<n;i++){
     const a=(Math.random()+Math.random()-1)*0.55; // Gaussian ≈ ±0.4 rad from centre
     const sc=(Math.random()+Math.random()-1)*0.18;
     const cosA=Math.cos(a),sinA=Math.sin(a),cosS=Math.cos(sc),sinS=Math.sin(sc)*(Math.random()<.5?1:-1);
     const ix=cosA*Ux+sinA*Vx, iy=cosA*Uy+sinA*Vy, iz=cosA*Uz+sinA*Vz;
     const r=80+Math.random()*28;
     p.push(r*(cosS*ix+sinS*nx), r*(cosS*iy+sinS*ny), r*(cosS*iz+sinS*nz));
     const w=0.55+Math.random()*0.45;
     c.push(w,w*0.80,w*0.40);
   }
   pts(p,c,0.13,0.65,THREE.AdditiveBlending);}

  // ── 5. Nebula patches (additive blended colour clouds) ──────
  const patches=[
    {cx: Ux*52, cy:Uy*52, cz:Uz*52, spread:20, col:[0.05,0.15,0.90], n:500},  // blue near centre
    {cx:-Ux*60, cy:-Uy*60, cz:-Uz*60, spread:18, col:[0.80,0.08,0.18], n:350}, // red anti-centre
    {cx: Vx*58, cy:Vy*58, cz:Vz*58, spread:22, col:[0.05,0.65,0.35], n:300},  // teal off-axis
    {cx:-Vx*50+Ux*30, cy:-Vy*50+Uy*30, cz:-Vz*50+Uz*30, spread:16, col:[0.55,0.05,0.75], n:250}, // purple
  ];
  for(const pp of patches){
    const pos=[],col=[];
    for(let i=0;i<pp.n;i++){
      const dx=(Math.random()+Math.random()-1)*pp.spread;
      const dy=(Math.random()+Math.random()-1)*pp.spread*0.6;
      const dz=(Math.random()+Math.random()-1)*pp.spread;
      pos.push(pp.cx+dx, pp.cy+dy, pp.cz+dz);
      const f=0.4+Math.random()*0.6;
      col.push(pp.col[0]*f, pp.col[1]*f, pp.col[2]*f);
    }
    pts(pos,col,1.6,0.09,THREE.AdditiveBlending);
  }
})();

// ─── Sky photo (equirectangular, replaces generated starfield on load) ────────

new THREE.TextureLoader().load('./sky.jpg',
  (tex)=>{
    tex.mapping=THREE.EquirectangularReflectionMapping;
    tex.colorSpace=THREE.SRGBColorSpace;
    scene.background=tex;
    scene.backgroundIntensity=0.14; // dim so the matrix space reads clearly
    scene.remove(starfieldGrp); // real photo is better — drop generated points
    hasSkyPhoto=true;
  },
  undefined,
  ()=>console.warn('sky.jpg failed to load — using generated starfield')
);

// ─── Floor ────────────────────────────────────────────────────────────────────


const floorMesh=new THREE.Mesh(new THREE.PlaneGeometry(20,20),new THREE.MeshBasicMaterial({visible:false,side:THREE.DoubleSide}));
floorMesh.rotation.x=-Math.PI/2;scene.add(floorMesh);
const reticle=new THREE.Mesh(new THREE.RingGeometry(0.12,0.18,32),new THREE.MeshBasicMaterial({color:0x44ffaa,side:THREE.DoubleSide}));
reticle.rotation.x=-Math.PI/2;reticle.visible=false;scene.add(reticle);

// ─── Info panel ───────────────────────────────────────────────────────────────

const PANEL_W=1024,PANEL_H=1600;
const panelCanvas=document.createElement('canvas');panelCanvas.width=PANEL_W;panelCanvas.height=PANEL_H;
const panelCtx=panelCanvas.getContext('2d');
const panelTex=new THREE.CanvasTexture(panelCanvas);
const panelMesh=new THREE.Mesh(new THREE.PlaneGeometry(1.8,2.81),new THREE.MeshBasicMaterial({map:panelTex,side:THREE.DoubleSide,transparent:true,depthWrite:false}));
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
  if(scenarioMode===1){
    if(t<0.01)return'Identity';if(t<=1)return'Stage 1 — V rotation in R³';
    if(t<=2)return'Stage 2 — Σ scale + collapse';return'Stage 3 — U rotation → image plane';
  }
  if(scenarioMode===2){
    if(t<0.01)return'Identity';if(t<=1)return'Stage 1 — V rotation in R²';
    if(t<=2)return'Stage 2 — Σ stretching';return'Stage 3 — U lift to R³';
  }
  if(scenarioMode===3){
    if(t<0.01)return'Original 3D cloud';if(t<=1)return'Stage 1 — Align to PC axes';
    if(t<=2)return'Stage 2 — Project to PC1-PC2';return'Stage 3 — Project to PC1 line';
  }
  if(scenarioMode===4){const cnt=s4Data?s4Data.planeCount:3;return`Least Squares \u2014 ${cnt} planes`;}
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
  if(matrixEditMode&&scenarioMode<3){drawMatrixEditPanel();return;}
  if(planeEditMode&&scenarioMode===4){drawEditPanel();return;}
  const ctx=panelCtx,W=PANEL_W,H=PANEL_H;
  ctx.clearRect(0,0,W,H);
  ctx.fillStyle='rgba(8,8,24,0.93)';drawRoundRect(ctx,0,0,W,H,28);ctx.fill();
  ctx.strokeStyle='rgba(80,120,255,0.45)';ctx.lineWidth=2;drawRoundRect(ctx,1,1,W-2,H-2,28);ctx.stroke();
  let y=46,IND=30;

  if(scenarioMode>0){
    ctx.fillStyle='#7799ff';ctx.font='bold 42px monospace';ctx.fillText(SCENARIO_NAMES[scenarioMode],IND,y);y+=42;
    if(scenarioMode!==4){
      ctx.fillStyle='#ffdd55';ctx.font='bold 30px monospace';ctx.fillText(`t = ${tParam.toFixed(2)}`,IND,y);
      ctx.fillStyle='#aaaaaa';ctx.font='24px monospace';ctx.fillText(stageName(tParam),IND+140,y);y+=36;
    } else {
      const s4t=(s4Data?s4Data.planeCount-3:0);
      ctx.fillStyle='#ffdd55';ctx.font='bold 30px monospace';ctx.fillText(`t = ${s4t.toFixed(2)}`,IND,y);
      ctx.fillStyle='#aaaaaa';ctx.font='24px monospace';ctx.fillText(stageName(tParam),IND+140,y);y+=36;
    }
    y=divider(ctx,y,W,IND);
    // ── Mode 1: 2×3 matrix + legend ─────────────────────────────────────────
    if(scenarioMode===1&&s1Data){
      ctx.fillStyle='#88bbff';ctx.font='bold 24px monospace';ctx.fillText('Matrix A (2×3):  R³ → R²',IND,y);y+=26;
      ctx.fillStyle='#bbccee';ctx.font='24px monospace';ctx.fillText('A =',IND,y);
      for(let r=0;r<2;r++){const row=s1Data.A[r].map(v=>String(v.toFixed(2)).padStart(6));ctx.fillText(`[ ${row.join('  ')} ]`,IND+60,y+r*29);}
      y+=2*29+10;y=divider(ctx,y,W,IND);
      ctx.fillStyle='#bbccee';ctx.font='24px monospace';
      for(const line of['Stage 1: rotate by V (3×3 right sing vecs)','Stage 2: Σ scaling + collapse z → plane','Stage 3: U rotation → image plane']){ctx.fillText(line,IND,y);y+=28;}
      y+=4;y=divider(ctx,y,W,IND);
      ctx.fillStyle='#88bbff';ctx.font='bold 26px monospace';ctx.fillText('SVD:  A = U · Σ · Vᵀ',IND,y);y+=30;
      ctx.fillStyle='#cccccc';ctx.font='24px monospace';
      ctx.fillText(`σ₁=${s1Data.svd.s[0].toFixed(3)},  σ₂=${s1Data.svd.s[1].toFixed(3)}`,IND,y);y+=28;
      {const[xU,xS,xV]=svdColX(W,IND);
      const sig=s1Data.svd.s;
      const Sigma1=[[sig[0],0,0],[0,sig[1],0]];
      const Vt1=s1Data.svd.V[0].map((_,c)=>s1Data.svd.V.map(r=>r[c]));
      const hU=panelMatBlock(ctx,'U (2×2)',s1Data.svd.U,xU,y,'#44cccc');
      panelMatBlock(ctx,'Σ (2×3)',Sigma1,xS,y,'#ffdd55');
      const hV=panelMatBlock(ctx,'Vᵀ (3×3)',Vt1,xV,y,'#cc88ff');
      y+=Math.max(hU,hV)+6;}
      y=divider(ctx,y,W,IND);
      ctx.fillStyle='#88bbff';ctx.font='bold 24px monospace';ctx.fillText('Legend',IND,y);y+=28;
      ctx.font='22px monospace';
      for(const[col,lbl] of[
        ['#00eeff','Original 3D points (ghost)'],
        ['#ff7722','Projected points'],
        ['#dddddd','Original bounding cube (white)'],
        ['#ff4444','Collapsing cube → image plane'],
        ['#4466ff','Image plane (SVD null space)'],
        [SV_COLORS[0],'v₁ — 1st right singular vec (σ₁)'],
        [SV_COLORS[1],'v₂ — 2nd right singular vec (σ₂)'],
      ]){swatch(ctx,IND,y,col,18,14);ctx.fillStyle='#cccccc';ctx.fillText(lbl,IND+26,y);y+=22;}
    }
    // ── Mode 2: 3×2 matrix + legend ─────────────────────────────────────────
    if(scenarioMode===2&&s2Data){
      ctx.fillStyle='#88bbff';ctx.font='bold 24px monospace';ctx.fillText('Matrix A (3×2):  R² → R³',IND,y);y+=26;
      ctx.fillStyle='#bbccee';ctx.font='24px monospace';ctx.fillText('A =',IND,y);
      for(let r=0;r<3;r++){const row=s2Data.A[r].map(v=>String(v.toFixed(2)).padStart(6));ctx.fillText(`[ ${row.join('  ')} ]`,IND+60,y+r*29);}
      y+=3*29+10;y=divider(ctx,y,W,IND);
      ctx.fillStyle='#bbccee';ctx.font='24px monospace';
      for(const line of['Stage 1: rotate by V (2×2 right sing vecs)','Stage 2: Σ stretching (σ₁, σ₂)','Stage 3: U lifts into 3D']){ctx.fillText(line,IND,y);y+=28;}
      y+=4;y=divider(ctx,y,W,IND);
      ctx.fillStyle='#88bbff';ctx.font='bold 26px monospace';ctx.fillText('SVD:  A = U · Σ · Vᵀ',IND,y);y+=30;
      ctx.fillStyle='#cccccc';ctx.font='24px monospace';
      ctx.fillText(`σ₁=${s2Data.svd.s[0].toFixed(3)},  σ₂=${s2Data.svd.s[1].toFixed(3)}`,IND,y);y+=28;
      {const[xU,xS,xV]=svdColX(W,IND);
      const sig=s2Data.svd.s;
      const Sigma2=[[sig[0],0],[0,sig[1]],[0,0]];
      const Vt2=s2Data.svd.V[0].map((_,c)=>s2Data.svd.V.map(r=>r[c]));
      const hU=panelMatBlock(ctx,'U (3×3)',s2Data.svd.U,xU,y,'#44cccc');
      panelMatBlock(ctx,'Σ (3×2)',Sigma2,xS,y,'#ffdd55');
      const hV=panelMatBlock(ctx,'Vᵀ (2×2)',Vt2,xV,y,'#cc88ff');
      y+=Math.max(hU,hV)+6;}
      y=divider(ctx,y,W,IND);
      ctx.fillStyle='#88bbff';ctx.font='bold 24px monospace';ctx.fillText('Legend',IND,y);y+=28;
      ctx.font='22px monospace';
      for(const[col,lbl] of[
        ['#00eeff','Original 2D input points (ghost)'],
        ['#ff7722','Lifted / transformed points'],
        ['#dddddd','Input square / output cube (white)'],
        ['#ff4444','Animating square → 3D (red)'],
        ['#ff4422','Input domain plane (z = 0)'],
        [SV_COLORS[0],'u₁ — 1st left singular vec (σ₁)'],
        [SV_COLORS[1],'u₂ — 2nd left singular vec (σ₂)'],
      ]){swatch(ctx,IND,y,col,18,14);ctx.fillStyle='#cccccc';ctx.fillText(lbl,IND+26,y);y+=22;}
    }
    // ── Mode 3: covariance / eigenvalues / variance + legend ─────────────────
    if(scenarioMode===3&&s3Data){
      const{Cov,evals}=s3Data.pca;
      ctx.fillStyle='#88bbff';ctx.font='bold 24px monospace';ctx.fillText('Covariance matrix (normalized):',IND,y);y+=26;
      ctx.fillStyle='#bbccee';ctx.font='24px monospace';ctx.fillText('Cov =',IND,y);
      for(let r=0;r<3;r++){const row=Cov[r].map(v=>String(v.toFixed(2)).padStart(6));ctx.fillText(`[ ${row.join('  ')} ]`,IND+72,y+r*28);}
      y+=3*28+10;y=divider(ctx,y,W,IND);
      const tot=evals[0]+evals[1]+evals[2]||1;
      ctx.fillStyle='#88bbff';ctx.font='bold 26px monospace';ctx.fillText('Eigenvalues:',IND,y);y+=30;
      ctx.fillStyle='#cccccc';ctx.font='24px monospace';
      ctx.fillText(`λ₁=${evals[0].toFixed(3)},  λ₂=${evals[1].toFixed(3)},  λ₃=${evals[2].toFixed(3)}`,IND,y);y+=28;
      ctx.fillStyle='#ffffaa';
      ctx.fillText(`Var 2D = ${((evals[0]+evals[1])/tot*100).toFixed(1)}%  (PC1+PC2)`,IND,y);y+=28;
      ctx.fillText(`Var 1D = ${(evals[0]/tot*100).toFixed(1)}%  (PC1 only)`,IND,y);y+=32;
      y=divider(ctx,y,W,IND);
      ctx.fillStyle='#bbccee';ctx.font='24px monospace';
      for(const line of['Stage 1: align to PC axes','Stage 2: project to PC1-PC2 plane','Stage 3: project to PC1 line']){ctx.fillText(line,IND,y);y+=28;}
      y=divider(ctx,y,W,IND);
      ctx.fillStyle='#88bbff';ctx.font='bold 24px monospace';ctx.fillText('Legend',IND,y);y+=28;
      ctx.font='22px monospace';
      for(const[col,lbl] of[
        ['#00eeff','Original 3D cloud (ghost)'],
        ['#ff7722','Aligned / projected points'],
        ['#dddddd','Bounding cube (reference)'],
        ['#22ff88','PC1-PC2 best-fit plane'],
        ['#ff4444','PC1 — largest variance axis'],
        ['#44ff88','PC2 — 2nd variance axis'],
        ['#4488ff','PC3 — smallest variance axis'],
      ]){swatch(ctx,IND,y,col,18,14);ctx.fillStyle='#cccccc';ctx.fillText(lbl,IND+26,y);y+=22;}
    }
    // ── Mode 4: LSE staged + legend ──────────────────────────────────────────
    if(scenarioMode===4&&s4Data){
      const cnt=s4Data.planeCount;
      const lseCur=[s4Data.lse3,s4Data.lse4,s4Data.lse5][cnt-3];
      const nPlanes=cnt;
      ctx.fillStyle='#bbccee';ctx.font='24px monospace';
      ctx.fillText(`${nPlanes} planes active`,IND,y);y+=28;
      ctx.fillText('LS minimizes \u03a3(dist\u00b2 to planes)',IND,y);y+=28;
      ctx.fillText('R-trig: add plane   L-trig: remove',IND,y);y+=28;
      y+=4;y=divider(ctx,y,W,IND);
      const x=lseCur.xLS;
      ctx.fillStyle='#88bbff';ctx.font='bold 26px monospace';ctx.fillText('LS Solution:',IND,y);y+=30;
      ctx.fillStyle='#ffdd55';ctx.font='24px monospace';
      ctx.fillText(`x=${x[0].toFixed(3)}, y=${x[1].toFixed(3)}, z=${x[2].toFixed(3)}`,IND,y);y+=28;
      ctx.fillStyle='#cccccc';
      ctx.fillText(`Residuals: ${lseCur.dists.map(d=>d.toFixed(3)).join(', ')}`,IND,y);y+=28;
      const rms4=Math.sqrt(lseCur.dists.reduce((s,d)=>s+d*d,0)/lseCur.dists.length);
      ctx.fillStyle='#ff8844';ctx.font='bold 24px monospace';
      ctx.fillText(`RMS Error: ${rms4.toFixed(4)}`,IND,y);y+=32;
      y=divider(ctx,y,W,IND);
      ctx.fillStyle='#88bbff';ctx.font='bold 24px monospace';ctx.fillText('Legend',IND,y);y+=28;
      ctx.font='22px monospace';
      for(const[col,lbl] of[
        ['#ffffcc','LS solution (animated sphere)'],
        ['#aaaaaa','Residual lines to planes'],
        ['#ff4444','Plane 1:  x + y + z = 3'],
        ['#44ff44','Plane 2:  x \u2212 y = 0'],
        ['#4488ff','Plane 3:  y \u2212 z = \u22121'],
        ['#ffaa22','Plane 4:  x \u2212 2z = \u22122  (stage 1+)'],
        ['#ff44ff','Plane 5:  2x + y = 4  (stage 2)'],
        ['#dddddd','Scene bounds (white cube)'],
      ]){swatch(ctx,IND,y,col,18,14);ctx.fillStyle='#cccccc';ctx.fillText(lbl,IND+26,y);y+=22;}
    }
    y=divider(ctx,y,W,IND);
    ctx.fillStyle='#7799ff';ctx.font='bold 24px monospace';ctx.fillText('VR Controls',IND,y);y+=28;
    ctx.font='21px monospace';
    const c2='#88aadd';
    for(const[dot,key,desc2] of[
      ['#ffdd55','R trigger','→ advance t (0→3)'],['#ffdd55','L trigger','→ reverse t (3→0)'],
      ['#ff8844','R grip','→ cycle 3×3 matrix'],['#44ffcc','L stick ←→','→ cycle scenario mode'],
      ['#44ffaa','L stick click','→ teleport'],['#44ffaa','R stick Y','→ zoom'],
      ['#ff88ff','A button','→ grab/rotate/throw'],['#ffaa44','B button (mode 4)','→ plane editor'],
      ['#aaaaaa','S key','→ cycle scenario (desktop)'],
    ]){
      ctx.fillStyle=dot;ctx.fillRect(IND,y-12,10,14);
      ctx.fillStyle='#ffffff';ctx.fillText(key,IND+16,y);
      ctx.fillStyle=c2;ctx.fillText(desc2,IND+170,y);y+=24;
    }
    panelTex.needsUpdate=true;return;
  }

  const preset=PRESETS[presetIdx];const A=mat0Custom;const svd=currentSVD;
  ctx.fillStyle='#7799ff';ctx.font='bold 42px monospace';ctx.fillText('3×3 SVD Transform',IND,y);y+=42;
  ctx.fillStyle='#88aadd';ctx.font='bold 26px monospace';ctx.fillText(`Preset: ${preset.name}`,IND,y);y+=30;
  ctx.fillStyle='#bbccee';ctx.font='24px monospace';ctx.fillText('A =',IND,y);
  for(let r=0;r<3;r++){const row=A[r].map(v=>String(v.toFixed(2)).padStart(6));ctx.fillText(`[ ${row.join('  ')} ]`,IND+60,y+r*29);}
  y+=3*29+8;y=divider(ctx,y,W,IND);
  ctx.fillStyle='#ffdd55';ctx.font='bold 30px monospace';ctx.fillText(`t = ${tParam.toFixed(2)}`,IND,y);
  ctx.fillStyle='#aaaaaa';ctx.font='26px monospace';ctx.fillText(stageName(tParam),IND+148,y);y+=38;
  y=divider(ctx,y,W,IND);
  ctx.fillStyle='#88bbff';ctx.font='bold 28px monospace';ctx.fillText('SVD:  A = U · Σ · Vᵀ',IND,y);y+=34;
  if(svd){
    const sv=[svd.Sigma[0][0],svd.Sigma[1][1],svd.Sigma[2][2]];
    ctx.fillStyle='#cccccc';ctx.font='24px monospace';
    ctx.fillText(`σ = [ ${sv.map(s=>s.toFixed(3)).join(',  ')} ]`,IND,y);y+=30;
    const[xU,xS,xV]=svdColX(W,IND);
    const Vt=svd.V[0].map((_,c)=>svd.V.map(r=>r[c]));
    const hU=panelMatBlock(ctx,'U (3×3)',svd.U,xU,y,'#44cccc');
    panelMatBlock(ctx,'Σ (3×3)',svd.Sigma,xS,y,'#ffdd55');
    panelMatBlock(ctx,'Vᵀ (3×3)',Vt,xV,y,'#cc88ff');
    y+=hU+6;
  }
  y=divider(ctx,y,W,IND);
  ctx.fillStyle='#88bbff';ctx.font='bold 28px monospace';ctx.fillText('Legend',IND,y);y+=32;
  ctx.font='23px monospace';
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
  ctx.fillStyle='#cc88ff';ctx.font='23px monospace';ctx.fillText('V rotation axis (stage 1)',IND+30,y);y+=27;
  ctx.setLineDash([8,5]);
  ctx.strokeStyle='#00cccc';ctx.lineWidth=3;ctx.beginPath();ctx.moveTo(IND,y-8);ctx.lineTo(IND+22,y-8);ctx.stroke();ctx.setLineDash([]);
  ctx.fillStyle='#44cccc';ctx.fillText('U rotation axis (stage 3)',IND+30,y);y+=32;
  y=divider(ctx,y,W,IND);
  ctx.fillStyle='#7799ff';ctx.font='bold 24px monospace';
  ctx.fillText('VR Controls',IND,y);y+=28;
  ctx.font='21px monospace';
  const ctrl2col='#88aadd';
  const ctrlRows=[
    ['#ffdd55','R trigger','→ advance t (0→3)'],
    ['#ffdd55','L trigger','→ reverse t (3→0)'],
    ['#ff8844','R grip','→ cycle matrix preset'],
    ['#44ffaa','L stick click','→ teleport'],
    ['#44ffaa','R stick Y','→ zoom in / out (0.1× – 2×)'],
    ['#ff88ff','A button','→ grab space: drag + rotate'],
    ['#ff88ff','A release','→ throw (momentum carry)'],
    ['#ffaa44','B button (mode 4)','→ open plane equation editor'],
    ['#44ffcc','L stick ←→','→ right=next / left=prev scenario'],
    ['#aaaaaa','S key','→ cycle scenario (desktop)'],
  ];
  for(const[dot,key,desc] of ctrlRows){
    ctx.fillStyle=dot;ctx.fillRect(IND,y-12,10,14);
    ctx.fillStyle='#ffffff';ctx.fillText(key,IND+16,y);
    ctx.fillStyle=ctrl2col;ctx.fillText(desc,IND+170,y);
    y+=24;
  }
  panelTex.needsUpdate=true;
}

// ─── Wrist HUD ────────────────────────────────────────────────────────────────

const wristCanvas=document.createElement('canvas');wristCanvas.width=512;wristCanvas.height=340;
const wristCtx=wristCanvas.getContext('2d');
const wristTex=new THREE.CanvasTexture(wristCanvas);
const wristMesh=new THREE.Mesh(
  new THREE.PlaneGeometry(0.22,0.146),
  new THREE.MeshBasicMaterial({map:wristTex,side:THREE.DoubleSide,transparent:true,depthWrite:false})
);
wristMesh.position.set(0,0.07,-0.03);
wristMesh.rotation.x=-Math.PI*0.38;
let wristHUDAttached=false;

function updateWristHUD(){
  const W=512,H=340,ctx=wristCtx,P=14;
  ctx.clearRect(0,0,W,H);
  ctx.fillStyle='rgba(8,10,28,0.95)';drawRoundRect(ctx,0,0,W,H,14);ctx.fill();
  ctx.strokeStyle='rgba(80,140,255,0.6)';ctx.lineWidth=2;drawRoundRect(ctx,1,1,W-2,H-2,14);ctx.stroke();
  const wDiv=()=>{ctx.strokeStyle='rgba(80,120,200,0.35)';ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(P,y+4);ctx.lineTo(W-P,y+4);ctx.stroke();y+=12;};
  let y=P;
  // ── Title ────────────────────────────────────────────────────────────────────
  const title=scenarioMode===0?'3×3 SVD Transform':SCENARIO_NAMES[scenarioMode];
  ctx.fillStyle='#7799ff';ctx.font='bold 26px monospace';ctx.fillText(title,P,y+22);y+=28;
  // t value + stage name
  if(scenarioMode!==4){
    ctx.fillStyle='#ffdd55';ctx.font='bold 21px monospace';ctx.fillText(`t = ${tParam.toFixed(2)}`,P,y+18);
    ctx.fillStyle='#aaaacc';ctx.font='15px monospace';ctx.fillText(stageName(tParam),P+108,y+18);y+=24;
  } else {
    const s4t=(s4Data?s4Data.planeCount-3:0);
    ctx.fillStyle='#ffdd55';ctx.font='bold 21px monospace';ctx.fillText(`t = ${s4t.toFixed(2)}`,P,y+18);
    ctx.fillStyle='#aaaacc';ctx.font='15px monospace';ctx.fillText(stageName(tParam),P+108,y+18);y+=24;
  }
  wDiv();
  // ── Main info (per scenario) ──────────────────────────────────────────────────
  ctx.font='19px monospace';
  if(scenarioMode===0&&currentSVD){
    const sv=[currentSVD.Sigma[0][0],currentSVD.Sigma[1][1],currentSVD.Sigma[2][2]];
    ctx.fillStyle='#88aadd';ctx.fillText(`Preset: ${PRESETS[presetIdx].name}`,P,y+16);y+=21;
    ctx.fillStyle='#cccccc';ctx.fillText('Decomposition:  A = U \u00b7 \u03a3 \u00b7 V\u1d40',P,y+16);y+=21;
    ctx.fillStyle='#aaccff';ctx.fillText(`\u03c3 = [ ${sv.map(s=>s.toFixed(2)).join(',  ')} ]`,P,y+16);y+=22;
  }
  if(scenarioMode===1&&s1Data){
    ctx.fillStyle='#aaccff';ctx.fillText('A = [ [1, 2, 0], [0, 1, -1] ]',P,y+16);y+=21;
    ctx.fillStyle='#cccccc';ctx.fillText(`\u03c3\u2081=${s1Data.svd.s[0].toFixed(2)}   \u03c3\u2082=${s1Data.svd.s[1].toFixed(2)}`,P,y+16);y+=21;
    ctx.fillStyle='#88aadd';ctx.fillText('Rank-2 projection:  R\u00b3 \u2192 R\u00b2',P,y+16);y+=22;
  }
  if(scenarioMode===2&&s2Data){
    ctx.fillStyle='#aaccff';ctx.fillText('A = [ [1,2], [0,1], [-1,0] ]',P,y+16);y+=21;
    ctx.fillStyle='#cccccc';ctx.fillText(`\u03c3\u2081=${s2Data.svd.s[0].toFixed(2)}   \u03c3\u2082=${s2Data.svd.s[1].toFixed(2)}`,P,y+16);y+=21;
    ctx.fillStyle='#88aadd';ctx.fillText('Rank-2 lifting:  R\u00b2 \u2192 R\u00b3',P,y+16);y+=22;
  }
  if(scenarioMode===3&&s3Data){
    const{evals}=s3Data.pca,tot=evals[0]+evals[1]+evals[2]||1;
    ctx.fillStyle='#aaccff';ctx.fillText(`\u03bb: ${evals.map(e=>e.toFixed(2)).join('   ')}`,P,y+16);y+=21;
    ctx.fillStyle='#ffffaa';ctx.fillText(`Var 2D=${((evals[0]+evals[1])/tot*100).toFixed(0)}%   Var 1D=${(evals[0]/tot*100).toFixed(0)}%`,P,y+16);y+=21;
    ctx.fillStyle='#88aadd';ctx.fillText('60-point pancake cloud  \u2192  PCA',P,y+16);y+=22;
  }
  if(scenarioMode===4&&s4Data){
    const cnt=s4Data.planeCount;
    const lseCur=[s4Data.lse3,s4Data.lse4,s4Data.lse5][cnt-3];
    const xls=lseCur.xLS;
    ctx.fillStyle='#cccccc';ctx.fillText(`${cnt} planes active`,P,y+16);y+=21;
    ctx.fillStyle='#ffdd55';ctx.fillText(`x=${xls[0].toFixed(2)}  y=${xls[1].toFixed(2)}  z=${xls[2].toFixed(2)}`,P,y+16);y+=21;
    ctx.fillStyle='#88aadd';ctx.fillText(`Res: ${lseCur.dists.map(d=>d.toFixed(2)).join('  ')}`,P,y+16);y+=21;
    const rms4w=Math.sqrt(lseCur.dists.reduce((s,d)=>s+d*d,0)/lseCur.dists.length);
    ctx.fillStyle='#ff8844';ctx.fillText(`RMS: ${rms4w.toFixed(4)}`,P,y+16);y+=22;
  }
  wDiv();
  // ── Steps ────────────────────────────────────────────────────────────────────
  ctx.fillStyle='#88bbff';ctx.font='bold 19px monospace';ctx.fillText('Steps:',P,y+15);y+=21;
  const steps={
    0:[['t:0\u21921','Rotate pts by V  (align to right sing vecs)'],
       ['t:1\u21922','Scale each axis by \u03a3  (singular stretching)'],
       ['t:2\u21923','Rotate by U  \u2192  arrives at full A\u00b7x']],
    1:[['t:0\u21921','Rotate 3D cloud by V  (right singular vecs)'],
       ['t:1\u21922','Scale axes by \u03a3 + collapse z \u2192 zero'],
       ['t:2\u21923','Apply U rotation  \u2192  land on image plane']],
    2:[['t:0\u21921','Rotate flat 2D square by V  (in-plane)'],
       ['t:1\u21922','Scale by \u03c3\u2081,\u03c3\u2082  (2D stretching)'],
       ['t:2\u21923','U lifts points from z=0 into 3D space']],
    3:[['t:0\u21921','Rotate cloud to align with PC axes'],
       ['t:1\u21922','Project out PC3  \u2192  flatten to 2D plane'],
       ['t:2\u21923','Project out PC2  \u2192  collapse to 1D line']],
    4:[['R-trig','Add a plane  (3 \u2192 4 \u2192 5)'],
       ['L-trig','Remove a plane  (5 \u2192 4 \u2192 3)'],
       ['\u2014','Sphere animates to new LS solution']],
  }[scenarioMode]||[];
  ctx.font='15px monospace';
  for(let i=0;i<steps.length;i++){
    const[tLabel,desc]=steps[i];
    const active=scenarioMode<4&&tParam>=i&&tParam<i+1;
    ctx.fillStyle=active?'#ffdd55':'#556688';ctx.fillText(tLabel,P,y+13);
    ctx.fillStyle=active?'#ffffff':'#99aabb';ctx.fillText(desc,P+72,y+13);y+=20;
  }
  wDiv();
  // ── Quick-reference tip ───────────────────────────────────────────────────────
  if(planeEditMode&&scenarioMode===4){
    ctx.fillStyle='#ffaa44';ctx.font='bold 16px monospace';
    ctx.fillText(`EDITING P${planeEditRow+1} \u00b7 ${'ABCD'[planeEditCol]}  = ${s4PlanesCustom[planeEditRow][planeEditCol].toFixed(1)}`,P,y+12);
  } else {
    ctx.fillStyle='#445566';ctx.font='13px monospace';
    ctx.fillText(scenarioMode===4?'R-trig: +plane   L-trig: \u2212plane   L-stick\u2194: mode'
      :'R-trig: fwd   L-trig: rev   L-stick\u2194: mode',P,y+12);
  }
  wristTex.needsUpdate=true;
}

// ─── Plane editor (VR, mode 4) ────────────────────────────────────────────────

let planeEditMode=false,planeEditRow=0,planeEditCol=0;
let editPrevStickX=false,editPrevStickY=false,editPrevLGrip=false;
let matrixEditMode=false,matEditRow=0,matEditCol=0;
let matEditPrevStickX=false,matEditPrevStickY=false,matEditPrevLGrip=false;

// ─── Plane grab mode (VR, mode 4) ─────────────────────────────────────────────
let vrGrabMode=false,prevXBtn=false;
let grabbedPlaneIdx=-1,hoverPlaneIdx=-1;
const planeGrabPrevPos=new THREE.Vector3(),planeGrabPrevQuat=new THREE.Quaternion();
let prevPlaneGrip=false,vrHandAnimT=0;

function buildVRHand(){
  const grp=new THREE.Group();
  const mat=new THREE.MeshStandardMaterial({color:0xffccaa,roughness:0.85,metalness:0.0});
  function box(w,h,d){return new THREE.Mesh(new THREE.BoxGeometry(w,h,d),mat);}
  // Palm — flat box, fingers extend in -Z from its front edge
  const palm=box(0.085,0.016,0.090);palm.position.set(0,-0.010,0.120);grp.add(palm);
  // Finger definitions: [x_offset, [proximal,middle,distal] lengths in metres]
  const fDefs=[
    [-0.030,[0.038,0.030,0.022]], // index
    [-0.010,[0.043,0.033,0.024]], // middle
    [ 0.010,[0.040,0.031,0.022]], // ring
    [ 0.030,[0.032,0.025,0.018]], // pinky
  ];
  const fingerChains=[];
  for(const[fx,[l0,l1,l2]] of fDefs){
    const j0=new THREE.Group();j0.position.set(fx,-0.002,0.075);grp.add(j0);
    const s0=box(0.018,0.016,l0);s0.position.z=-l0/2;j0.add(s0);
    const j1=new THREE.Group();j1.position.z=-l0;j0.add(j1);
    const s1=box(0.016,0.015,l1);s1.position.z=-l1/2;j1.add(s1);
    const j2=new THREE.Group();j2.position.z=-l1;j1.add(j2);
    const s2=box(0.015,0.014,l2);s2.position.z=-l2/2;j2.add(s2);
    fingerChains.push([j0,j1,j2]);
  }
  // Thumb — 2 segments angled outward from palm side
  const tj0=new THREE.Group();tj0.position.set(-0.050,-0.005,0.108);
  tj0.rotation.z=0.42;tj0.rotation.y=-0.28;grp.add(tj0);
  const ts0=box(0.020,0.018,0.035);ts0.position.z=-0.0175;tj0.add(ts0);
  const tj1=new THREE.Group();tj1.position.z=-0.035;tj0.add(tj1);
  const ts1=box(0.018,0.016,0.028);ts1.position.z=-0.014;tj1.add(ts1);
  const thumbChain=[tj0,tj1];
  return{grp,fingerChains,thumbChain};
}

function updateVRHandCurl(fingerChains,thumbChain,t){
  const c=[1.22,1.48,1.22]; // knuckle, middle, tip angles
  for(const chain of fingerChains)for(let j=0;j<3;j++)chain[j].rotation.x=t*c[j];
  thumbChain[0].rotation.x=t*0.75;thumbChain[1].rotation.x=t*0.55;
}

const vrHand=buildVRHand();scene.add(vrHand.grp);vrHand.grp.visible=false;
const _vrHandOffset=new THREE.Quaternion().setFromEuler(new THREE.Euler(0,0,Math.PI));

function drawEditPanel(){
  const ctx=panelCtx,W=PANEL_W,H=PANEL_H,IND=30;
  ctx.clearRect(0,0,W,H);
  ctx.fillStyle='rgba(8,8,24,0.97)';drawRoundRect(ctx,0,0,W,H,28);ctx.fill();
  ctx.strokeStyle='rgba(255,160,60,0.7)';ctx.lineWidth=3;drawRoundRect(ctx,2,2,W-4,H-4,28);ctx.stroke();
  let y=58;
  ctx.fillStyle='#ffaa44';ctx.font='bold 50px monospace';ctx.fillText('PLANE  EDITOR',IND,y);y+=54;
  ctx.fillStyle='#888888';ctx.font='24px monospace';ctx.fillText('Ax + By + Cz = D',IND,y);y+=34;
  y=divider(ctx,y,W,IND);
  // Column headers
  const CX=[150,350,550,750];
  ctx.fillStyle='#aaaaaa';ctx.font='bold 29px monospace';
  ctx.fillText('Plane',IND,y);
  for(let c=0;c<4;c++)ctx.fillText('ABCD'[c],CX[c],y);
  y+=8;y=divider(ctx,y,W,IND);
  // Rows
  for(let r=0;r<5;r++){
    const selRow=r===planeEditRow;
    if(selRow){ctx.fillStyle='rgba(255,160,60,0.10)';ctx.fillRect(IND-6,y-26,W-IND,42);}
    ctx.fillStyle=selRow?'#ffaa44':'#444455';ctx.font='bold 29px monospace';
    ctx.fillText(selRow?'\u25b6':' ',IND,y);
    ctx.fillStyle=S4_CSS_COLORS[r];ctx.font='bold 29px monospace';
    ctx.fillText(`P${r+1}`,IND+28,y);
    for(let c=0;c<4;c++){
      const selCell=selRow&&c===planeEditCol;
      const val=s4PlanesCustom[r][c];
      const str=val.toFixed(1);
      if(selCell){
        const tw=ctx.measureText(str).width;
        ctx.fillStyle='rgba(255,220,80,0.28)';ctx.fillRect(CX[c]-7,y-24,tw+14,32);
        ctx.fillStyle='#ffee44';ctx.font='bold 29px monospace';
      } else {
        ctx.fillStyle=selRow?'#cccccc':'#778899';ctx.font='29px monospace';
      }
      ctx.fillText(str,CX[c],y);
    }
    y+=42;
  }
  y+=6;y=divider(ctx,y,W,IND);
  ctx.fillStyle='#aaddff';ctx.font='bold 26px monospace';ctx.fillText('Controls',IND,y);y+=32;
  ctx.font='23px monospace';
  for(const[dot,key,desc] of[
    ['#ffaa44','L stick \u2191\u2193','  +0.5 / \u22120.5 on selected value'],
    ['#ffaa44','L stick \u2190\u2192','  select A / B / C / D column'],
    ['#ffaa44','L grip',  '  cycle to next plane (P1\u2192P5)'],
    ['#ff88ff','B button','  exit editor'],
  ]){
    ctx.fillStyle=dot;ctx.fillRect(IND,y-16,10,18);
    ctx.fillStyle='#ffffff';ctx.fillText(key,IND+18,y);
    ctx.fillStyle='#88aadd';ctx.fillText(desc,IND+290,y);y+=28;
  }
  panelTex.needsUpdate=true;
}

// ─── Shared helper: draw a labeled matrix block on a panel canvas ─────────────
function panelMatBlock(ctx,label,m,bx,by,color){
  const RH=25,FSV=19;
  ctx.fillStyle=color;ctx.font='bold 21px monospace';ctx.fillText(label,bx,by);
  let ly=by+27;
  ctx.font=`${FSV}px monospace`;
  for(let r=0;r<m.length;r++){
    const entries=m[r].map(v=>v.toFixed(2).padStart(6));
    ctx.fillStyle=color;ctx.fillText(`[${entries.join('')}]`,bx,ly);ly+=RH;
  }
  return ly-by;
}
function svdColX(W,IND){
  const span=(W-2*IND)/3;
  return[IND,Math.round(IND+span),Math.round(IND+2*span)];
}

// ─── Matrix editor panel (VR, modes 0-2) ──────────────────────────────────────

function drawMatrixEditPanel(){
  const ctx=panelCtx,W=PANEL_W,H=PANEL_H,IND=22;
  ctx.clearRect(0,0,W,H);
  ctx.fillStyle='rgba(8,8,24,0.97)';drawRoundRect(ctx,0,0,W,H,28);ctx.fill();
  ctx.strokeStyle='rgba(80,160,255,0.65)';ctx.lineWidth=3;drawRoundRect(ctx,2,2,W-4,H-4,28);ctx.stroke();
  let y=60;
  // ── Title ─────────────────────────────────────────────────────────────────────
  const titles=['MATRIX EDITOR  3×3','MATRIX EDITOR  2×3','MATRIX EDITOR  3×2'];
  ctx.fillStyle='#aaddff';ctx.font='bold 48px monospace';ctx.fillText(titles[scenarioMode],IND,y);y+=60;
  const mat=scenarioMode===0?mat0Custom:scenarioMode===1?mat1Custom:mat2Custom;
  const nRows=mat.length,nCols=mat[0].length;
  // ── Editable Matrix A ─────────────────────────────────────────────────────────
  ctx.fillStyle='#88bbff';ctx.font='bold 28px monospace';ctx.fillText('Matrix A  (editable):',IND,y);y+=36;
  const MAT_X=IND+70;const COL_W=Math.floor((W-MAT_X-IND)/nCols);
  ctx.fillStyle='#888888';ctx.font='24px monospace';
  for(let c=0;c<nCols;c++)ctx.fillText(`col ${c}`,MAT_X+c*COL_W,y);
  y+=8;y=divider(ctx,y,W,IND);
  for(let r=0;r<nRows;r++){
    const selRow=r===matEditRow;
    if(selRow){ctx.fillStyle='rgba(80,160,255,0.14)';ctx.fillRect(IND-4,y-30,W-IND+4,48);}
    ctx.fillStyle=selRow?'#aaddff':'#445566';ctx.font='bold 30px monospace';ctx.fillText(selRow?'▶':' ',IND,y);
    ctx.fillStyle='#bbccee';ctx.fillText(`r${r}`,IND+32,y);
    for(let c=0;c<nCols;c++){
      const selCell=selRow&&c===matEditCol;
      const str=mat[r][c].toFixed(2);
      ctx.font='bold 30px monospace';
      if(selCell){
        const tw=ctx.measureText(str).width;
        ctx.fillStyle='rgba(80,200,255,0.30)';ctx.fillRect(MAT_X+c*COL_W-6,y-28,tw+12,36);
        ctx.fillStyle='#44eeff';
      } else {
        ctx.fillStyle=selRow?'#cccccc':'#778899';
      }
      ctx.fillText(str,MAT_X+c*COL_W,y);
    }
    y+=48;
  }
  y+=8;y=divider(ctx,y,W,IND);
  // ── SVD: A = U · Σ · Vᵀ ──────────────────────────────────────────────────────
  ctx.fillStyle='#88bbff';ctx.font='bold 28px monospace';ctx.fillText('SVD:  A = U · Σ · Vᵀ',IND,y);y+=38;
  // helper: draw a labeled matrix block, returns height used
  const RH=28,FSV=21,LH=30;
  const xU=IND,xS=Math.round(IND+(W-2*IND)/3),xV=Math.round(IND+2*(W-2*IND)/3);
  function drawMatBlock(label,m,bx,by,color){
    ctx.fillStyle=color;ctx.font=`bold 23px monospace`;ctx.fillText(label,bx,by);
    let ly=by+LH;
    ctx.font=`${FSV}px monospace`;
    for(let r=0;r<m.length;r++){
      const entries=m[r].map(v=>v.toFixed(2).padStart(6));
      ctx.fillStyle=color;ctx.fillText(`[${entries.join('')}]`,bx,ly);ly+=RH;
    }
    return ly-by;
  }
  let svdH=0;
  if(scenarioMode===0&&currentSVD){
    const Vt=currentSVD.V[0].map((_,c)=>currentSVD.V.map(r=>r[c]));
    const hU=drawMatBlock('U (3×3)',currentSVD.U,xU,y,'#44cccc');
    drawMatBlock('Σ (3×3)',currentSVD.Sigma,xS,y,'#ffdd55');
    drawMatBlock('Vᵀ (3×3)',Vt,xV,y,'#cc88ff');
    svdH=hU;
  } else if(scenarioMode===1&&s1Data){
    const sig=s1Data.svd.s;
    const Sigma1=[[sig[0],0,0],[0,sig[1],0]];
    const Vt1=s1Data.svd.V[0].map((_,c)=>s1Data.svd.V.map(r=>r[c]));
    const hU=drawMatBlock('U (2×2)',s1Data.svd.U,xU,y,'#44cccc');
    drawMatBlock('Σ (2×3)',Sigma1,xS,y,'#ffdd55');
    const hV=drawMatBlock('Vᵀ (3×3)',Vt1,xV,y,'#cc88ff');
    svdH=Math.max(hU,hV);
  } else if(scenarioMode===2&&s2Data){
    const sig=s2Data.svd.s;
    const Sigma2=[[sig[0],0],[0,sig[1]],[0,0]];
    const Vt2=s2Data.svd.V[0].map((_,c)=>s2Data.svd.V.map(r=>r[c]));
    const hU=drawMatBlock('U (3×3)',s2Data.svd.U,xU,y,'#44cccc');
    drawMatBlock('Σ (3×2)',Sigma2,xS,y,'#ffdd55');
    drawMatBlock('Vᵀ (2×2)',Vt2,xV,y,'#cc88ff');
    svdH=hU;
  }
  y+=svdH+10;y=divider(ctx,y,W,IND);
  // ── Controls ──────────────────────────────────────────────────────────────────
  ctx.fillStyle='#aaddff';ctx.font='bold 28px monospace';ctx.fillText('Controls',IND,y);y+=36;
  ctx.font='24px monospace';
  for(const[dot,key,desc] of[
    ['#aaddff','L stick ↑↓','  +0.1 / −0.1 on selected value'],
    ['#aaddff','L stick ←→','  select column'],
    ['#aaddff','L grip',    '  cycle to next row'],
    ['#ff88ff','B button',  '  exit editor (matrix kept)'],
  ]){
    ctx.fillStyle=dot;ctx.fillRect(IND,y-18,10,20);
    ctx.fillStyle='#ffffff';ctx.fillText(key,IND+18,y);
    ctx.fillStyle='#88aadd';ctx.fillText(desc,IND+310,y);y+=32;
  }
  panelTex.needsUpdate=true;
}

// ─── Desktop HUD ──────────────────────────────────────────────────────────────

const hud=document.createElement('div');
Object.assign(hud.style,{position:'absolute',top:'12px',left:'12px',color:'#fff',
  fontFamily:'monospace',fontSize:'15px',background:'rgba(0,0,0,0.6)',
  padding:'10px 16px',borderRadius:'8px',lineHeight:'1.8',pointerEvents:'none'});
document.body.appendChild(hud);

function updateHUD(){
  const modeName=scenarioMode===0?`Matrix: <b>${PRESETS[presetIdx].name}</b> [1/2/3|G]`
    :`Scenario: <b>${SCENARIO_NAMES[scenarioMode]}</b> [S to cycle]`;
  const s4t=s4Data?s4Data.planeCount-3:0;
  const tLine=scenarioMode!==4
    ?`t = <b>${tParam.toFixed(2)}</b> / 3.00 &nbsp; <b>${stageName(tParam)}</b><br>`
    :`t = <b>${s4t.toFixed(2)}</b> &nbsp; <b>${stageName(tParam)}</b><br>`;
  const ctrlHint=scenarioMode!==4
    ?`← →=scrub &nbsp;|&nbsp; S=scenario &nbsp;|&nbsp; G=matrix`
    :vrGrabMode?`R-grip: grab plane &nbsp;|&nbsp; move/rotate &nbsp;|&nbsp; release X: apply`
    :planeEditMode?`L-stick ↑↓: ±0.5 &nbsp;|&nbsp; L-stick ←→: column &nbsp;|&nbsp; L-grip: plane &nbsp;|&nbsp; B: exit editor`
    :`← →=add/remove plane &nbsp;|&nbsp; B=plane editor &nbsp;|&nbsp; X=grab planes`;
  hud.innerHTML=`${modeName}<br>${tLine}Zoom: <b>${rootScale.toFixed(2)}x</b><br>`+
    `<span style="color:#aaa;font-size:12px">${ctrlHint}</span>`;
  if(s4InputPanel)s4InputPanel.style.display=(scenarioMode===4&&!renderer.xr.isPresenting)?'block':'none';
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

// ─── Plane-plane intersection helper ─────────────────────────────────────────

function computeIntersectLine(p1,p2,extent){
  const[a1,b1,c1,d1]=p1,[a2,b2,c2,d2]=p2;
  // Direction = n1 × n2
  const vx=b1*c2-c1*b2,vy=c1*a2-a1*c2,vz=a1*b2-b1*a2;
  const vLen=Math.sqrt(vx*vx+vy*vy+vz*vz);
  if(vLen<1e-10)return null;
  const dx=vx/vLen,dy=vy/vLen,dz=vz/vLen;
  // Find point on line: zero out the coord where |dir| is largest, solve 2×2
  const ax=Math.abs(dx),ay=Math.abs(dy),az=Math.abs(dz);
  let x0=0,y0=0,z0=0;
  if(ax>=ay&&ax>=az){
    const det=b1*c2-b2*c1;if(Math.abs(det)<1e-10)return null;
    y0=(-d1*c2+c1*d2)/det;z0=(-b1*d2+b2*d1)/det;
  }else if(ay>=ax&&ay>=az){
    const det=a1*c2-a2*c1;if(Math.abs(det)<1e-10)return null;
    x0=(-d1*c2+c1*d2)/det;z0=(-a1*d2+a2*d1)/det;
  }else{
    const det=a1*b2-a2*b1;if(Math.abs(det)<1e-10)return null;
    x0=(-d1*b2+b1*d2)/det;y0=(-a1*d2+a2*d1)/det;
  }
  return{
    start:[x0-dx*extent,y0-dy*extent,z0-dz*extent],
    end:  [x0+dx*extent,y0+dy*extent,z0+dz*extent],
  };
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
function makeLabel(text,colorStr,size='normal'){
  const cfg={
    normal:{w:320,h:72,  font:'bold 28px monospace',tx:160,ty:50,sx:0.55,sy:0.13},
    big:   {w:768,h:110, font:'bold 52px monospace',tx:384,ty:78,sx:1.30,sy:0.19},
    axis:  {w:80, h:80,  font:'bold 56px monospace',tx:40, ty:60,sx:0.22,sy:0.22},
  }[size]||{w:256,h:64,font:'bold 26px monospace',tx:128,ty:44,sx:0.44,sy:0.11};
  const c=document.createElement('canvas');c.width=cfg.w;c.height=cfg.h;
  const ctx=c.getContext('2d');
  ctx.font=cfg.font;
  ctx.shadowColor=colorStr;ctx.shadowBlur=12;
  ctx.fillStyle=colorStr;ctx.textAlign='center';ctx.fillText(text,cfg.tx,cfg.ty);
  const tex=new THREE.CanvasTexture(c);
  const mat=new THREE.SpriteMaterial({map:tex,transparent:true,depthWrite:false});
  const sprite=new THREE.Sprite(mat);sprite.scale.set(cfg.sx,cfg.sy,1);return sprite;
}

function addAxisLabels(){
  for(const[t,x,y,z,col] of[['X',2.05,0,0,'#ff4444'],['Y',0,2.05,0,'#44ff44'],['Z',0,0,2.05,'#4488ff']]){
    const lbl=makeLabel(t,col,'axis');lbl.position.set(x,y,z);root.add(lbl);
  }
}

// ─── Trail system ─────────────────────────────────────────────────────────────

const TRAIL_N=10;
const TRAIL_MAX_OP=0.30;
const TRAIL_DECAY=2.2; // exponential decay rate (higher = faster fade)
let trailPts=[],trailBufs=[],trailAges=[],trailWriteIdx=0,lastTrailT=-99;

function initTrails(){
  trailPts.forEach(p=>root.remove(p));trailPts=[];trailBufs=[];trailAges=[];
  for(let i=0;i<TRAIL_N;i++){
    const buf=new Float32Array(30*3);
    const geo=new THREE.BufferGeometry();geo.setAttribute('position',new THREE.Float32BufferAttribute(buf.slice(),3));
    const mat=new THREE.PointsMaterial({size:0.042,transparent:true,opacity:0,color:0xff9944,sizeAttenuation:true,depthWrite:false});
    const pts=new THREE.Points(geo,mat);pts.visible=false;
    root.add(pts);trailPts.push(pts);trailBufs.push(buf);trailAges.push(-1);
  }
  trailWriteIdx=0;lastTrailT=-99;
}

function pushTrail(tPts){
  if(Math.abs(tParam-lastTrailT)<0.025)return;
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

let tParam=0,presetIdx=0,currentSVD=null,points=[],cubeCorners=[];
let origIM,transIM,cubeOL,cubeTL,dispL,axisVL,axisUL;
let svLabels=[];

// ─── Scenario mode ────────────────────────────────────────────────────────────
let scenarioMode=0;
const SCENARIO_NAMES=['3×3 SVD','2×3 Projection (R³→R²)','3×2 Lifting (R²→R³)','3D PCA','Least Squares'];
let prevLeftStickTriggered=false;
let s1Data=null,s2Data=null,s3Data=null,s4Data=null;

// ─── Mode 4 sphere trail ───────────────────────────────────────────────────────
const S4_TRAIL_LEN=48;
const s4TrailPosArr=new Float32Array(S4_TRAIL_LEN*3);
const s4TrailColArr=new Float32Array(S4_TRAIL_LEN*3);
const s4TrailPosBuf=new THREE.BufferAttribute(s4TrailPosArr,3);
const s4TrailColBuf=new THREE.BufferAttribute(s4TrailColArr,3);
const s4TrailGeo=new THREE.BufferGeometry();
s4TrailGeo.setAttribute('position',s4TrailPosBuf);
s4TrailGeo.setAttribute('color',s4TrailColBuf);
const s4TrailMat=new THREE.LineBasicMaterial({vertexColors:true,transparent:true,opacity:0,depthWrite:false});
const s4TrailLine=new THREE.Line(s4TrailGeo,s4TrailMat);
s4TrailLine.frustumCulled=false;s4TrailLine.visible=false;
let s4TrailData=[],s4TrailFadeAge=-1;

function s4AddTrailPoint(pos){
  s4TrailData.push([pos.x,pos.y,pos.z]);
  if(s4TrailData.length>S4_TRAIL_LEN)s4TrailData.shift();
  const n=s4TrailData.length;
  for(let i=0;i<n;i++){
    const t=n>1?i/(n-1):1;
    s4TrailPosArr[i*3]=s4TrailData[i][0];s4TrailPosArr[i*3+1]=s4TrailData[i][1];s4TrailPosArr[i*3+2]=s4TrailData[i][2];
    s4TrailColArr[i*3]=t;s4TrailColArr[i*3+1]=t*0.55;s4TrailColArr[i*3+2]=t*0.05;
  }
  s4TrailPosBuf.needsUpdate=true;s4TrailColBuf.needsUpdate=true;
  s4TrailGeo.setDrawRange(0,n);
  s4TrailMat.opacity=1.0;s4TrailLine.visible=n>=2;s4TrailFadeAge=-1;
}

// ─── Mode 4 glow sprite ────────────────────────────────────────────────────────
const _s4GlowCv=document.createElement('canvas');_s4GlowCv.width=_s4GlowCv.height=64;
const _s4GlowCx=_s4GlowCv.getContext('2d');
const _s4Gr=_s4GlowCx.createRadialGradient(32,32,0,32,32,32);
_s4Gr.addColorStop(0,'rgba(255,230,140,1)');_s4Gr.addColorStop(0.2,'rgba(255,160,50,0.6)');
_s4Gr.addColorStop(0.55,'rgba(255,90,10,0.18)');_s4Gr.addColorStop(1,'rgba(0,0,0,0)');
_s4GlowCx.fillStyle=_s4Gr;_s4GlowCx.fillRect(0,0,64,64);
const s4GlowMat=new THREE.SpriteMaterial({map:new THREE.CanvasTexture(_s4GlowCv),transparent:true,depthWrite:false,blending:THREE.AdditiveBlending,opacity:0.5});
const s4GlowSprite=new THREE.Sprite(s4GlowMat);
s4GlowSprite.scale.setScalar(0.55);s4GlowSprite.visible=false;

// ─── Rebuild scene ────────────────────────────────────────────────────────────

function rebuildScene(speak=false){
  if(scenarioMode===1){buildScenario1(speak);return;}
  if(scenarioMode===2){buildScenario2(speak);return;}
  if(scenarioMode===3){buildScenario3(speak);return;}
  if(scenarioMode===4){buildScenario4(speak);return;}
  root.clear();svLabels=[];
  if(speak)speakText(`Matrix: ${PRESETS[presetIdx].name}`);
  root.add(new THREE.AxesHelper(1.8));
  addAxisLabels();
  const titleLbl=makeLabel('3×3 SVD Transform','#88aaff','big');
  titleLbl.position.set(0,2.65,0);root.add(titleLbl);
  const presetLbl=makeLabel(`Preset: ${PRESETS[presetIdx].name}`,'#aabbdd');
  presetLbl.position.set(0,2.38,0);root.add(presetLbl);

  const A=mat0Custom;
  currentSVD=makeRotationalSVD(A);
  points=genPoints(30,42);
  cubeCorners=buildCubeCorners(points);

  // InstancedMesh for original (ghost) and transformed points
  const sGeo=new THREE.SphereGeometry(0.05,8,6);
  origIM=new THREE.InstancedMesh(sGeo,new THREE.MeshLambertMaterial({color:0x00eeff,transparent:true,opacity:0.55}),points.length);
  transIM=new THREE.InstancedMesh(sGeo,new THREE.MeshLambertMaterial({color:0xff7722}),points.length);
  for(let i=0;i<points.length;i++){
    dummy.position.set(...points[i]);dummy.updateMatrix();
    origIM.setMatrixAt(i,dummy.matrix);transIM.setMatrixAt(i,dummy.matrix);
  }
  origIM.instanceMatrix.needsUpdate=true;
  transIM.instanceMatrix.needsUpdate=true;
  root.add(origIM);root.add(transIM);

  // Cube wireframes + displacement lines
  cubeOL=makeSegs(cubePosArray(cubeCorners),matCO);root.add(cubeOL);
  cubeTL=makeSegs(cubePosArray(cubeCorners),matCT);root.add(cubeTL);
  dispL =makeSegs(dispPosArray(points,points),matD);root.add(dispL);

  // Deforming 3D grid
  buildGridBase();
  const gridMat=new THREE.LineBasicMaterial({color:0x1a3a8a,transparent:true,opacity:0.50});
  gridMeshX=makeSegs(gridBaseX,gridMat);
  gridMeshY=makeSegs(gridBaseY,gridMat);
  gridMeshZ=makeSegs(gridBaseZ,gridMat);
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
  if(pulseRing){root.remove(pulseRing);pulseRing=null;}pulseAge=-1;lastTFloor=0;lastSpokenStage=-1;
  initTrails();
  updateSceneForT();updatePanel();updateHUD();updateWristHUD();
}

// ─── Update scene for current t ───────────────────────────────────────────────

function updateSceneForT(){
  if(scenarioMode===1)return updateScenario1();
  if(scenarioMode===2)return updateScenario2();
  if(scenarioMode===3)return updateScenario3();
  if(scenarioMode===4)return updateScenario4();
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
  if(!hasSkyPhoto)scene.background=bg;

  return tPts;
}

// ─── Scenario helpers ─────────────────────────────────────────────────────────

function makeSemiPlane(normal3,color,opacity,size){
  const m=new THREE.Mesh(new THREE.PlaneGeometry(size,size),
    new THREE.MeshBasicMaterial({color,transparent:true,opacity,side:THREE.DoubleSide,depthWrite:false}));
  const n=new THREE.Vector3(normal3[0],normal3[1],normal3[2]).normalize();
  const up=Math.abs(n.y)>0.99?new THREE.Vector3(1,0,0):new THREE.Vector3(0,1,0);
  m.quaternion.setFromUnitVectors(new THREE.Vector3(0,0,1),n);
  return m;
}

function makePointCloud(pts,color,opacity){
  const sGeo=new THREE.SphereGeometry(0.05,8,6);
  const im=new THREE.InstancedMesh(sGeo,
    new THREE.MeshLambertMaterial({color,transparent:opacity<1,opacity}),pts.length);
  for(let i=0;i<pts.length;i++){dummy.position.set(...pts[i]);dummy.updateMatrix();im.setMatrixAt(i,dummy.matrix);}
  im.instanceMatrix.needsUpdate=true;return im;
}

function updateIM(im,pts){
  for(let i=0;i<pts.length;i++){dummy.position.set(...pts[i]);dummy.updateMatrix();im.setMatrixAt(i,dummy.matrix);}
  im.instanceMatrix.needsUpdate=true;
}

function sceneCommon(speak,name){
  root.clear();svLabels=[];
  root.add(new THREE.AxesHelper(1.8));
  addAxisLabels();
  if(speak)speakText(name);
  if(pulseRing){root.remove(pulseRing);pulseRing=null;}
  pulseAge=-1;lastTFloor=0;lastSpokenStage=-1;
}

// ─── Scenario 1: 2×3 Projection (R³→R²) ─────────────────────────────────────

function buildScenario1(speak){
  sceneCommon(speak,SCENARIO_NAMES[1]);
  const A=mat1Custom,svd=svd2x3(A);
  const pts=genPoints(30,7);
  const V=svd.V;
  // Image plane: normal = V[:,2] (third right singular vector)
  root.add(makeSemiPlane([V[0][2],V[1][2],V[2][2]],0x4466ff,0.18,5));
  // Original (cyan ghost) + transformed (orange)
  const oIM=makePointCloud(pts,0x00eeff,0.45);
  const tIM=makePointCloud(pts,0xff7722,1.0);
  root.add(oIM);root.add(tIM);
  // Singular vector arrows (V columns)
  for(let i=0;i<2;i++){
    const d=[V[0][i],V[1][i],V[2][i]],len=Math.max(0.25,svd.s[i]*0.75);
    root.add(new THREE.ArrowHelper(new THREE.Vector3(...d).normalize(),new THREE.Vector3(),len,SV_HEX[i],len*0.25,len*0.13));
    const lbl=makeLabel(`v${i+1}  σ=${svd.s[i].toFixed(2)}`,SV_COLORS[i]);
    lbl.position.set(d[0]*len*1.4,d[1]*len*1.4,d[2]*len*1.4);root.add(lbl);
  }
  const lbl=makeLabel('2×3 Projection  R³→R²','#88aaff','big');lbl.position.set(0,2.5,0);root.add(lbl);
  // White cube (original 3D) + red cube (collapses to 2D plane)
  const cubeC=buildCubeCorners(pts);
  const cubeW=makeSegs(cubePosArray(cubeC),matCO);
  const cubeR=makeSegs(cubePosArray(cubeC),matCT);
  root.add(cubeW);root.add(cubeR);
  s1Data={svd,pts,tIM,cubeC,cubeR,A};
  initTrails();updateScenario1();updatePanel();updateHUD();updateWristHUD();
}

function updateScenario1(){
  if(!s1Data)return[];
  const tPts=pathScen1(s1Data.pts,tParam,s1Data.svd);
  updateIM(s1Data.tIM,tPts);
  const tCube=pathScen1(s1Data.cubeC,tParam,s1Data.svd);
  setLineSegs(s1Data.cubeR,cubePosArray(tCube));
  return tPts;
}

// ─── Scenario 2: 3×2 Lifting (R²→R³) ────────────────────────────────────────

function buildScenario2(speak){
  sceneCommon(speak,SCENARIO_NAMES[2]);
  const A=mat2Custom,svd=svd3x2(A);
  const pts2d=genPoints2D(30,7);
  const pts3d=pts2d.map(p=>[p[0],p[1],0]);
  // Domain plane (xy, z=0)
  root.add(makeSemiPlane([0,0,1],0xff4422,0.15,5));
  const oIM=makePointCloud(pts3d,0x00eeff,0.45);
  const tIM=makePointCloud(pts3d,0xff7722,1.0);
  root.add(oIM);root.add(tIM);
  // U singular vector arrows (left singular vectors — columns of U)
  const U=svd.U;
  for(let i=0;i<2;i++){
    const d=[U[0][i],U[1][i],U[2][i]],len=Math.max(0.25,svd.s[i]*0.75);
    root.add(new THREE.ArrowHelper(new THREE.Vector3(...d).normalize(),new THREE.Vector3(),len,SV_HEX[i],len*0.25,len*0.13));
    const lbl=makeLabel(`u${i+1}  σ=${svd.s[i].toFixed(2)}`,SV_COLORS[i]);
    lbl.position.set(d[0]*len*1.4,d[1]*len*1.4,d[2]*len*1.4);root.add(lbl);
  }
  const lbl=makeLabel('3×2 Lifting  R²→R³','#88aaff','big');lbl.position.set(0,2.5,0);root.add(lbl);
  // White cube: static reference for the 3D output space
  const ptsFull=pathScen2(pts2d,3.0,svd);
  root.add(makeSegs(cubePosArray(buildCubeCorners(ptsFull)),matCO));
  // White square (flat 2D input) + red square (lifts into 3D)
  const squareC=buildBoundingSquare2D(pts2d);
  const squareW=makeSegs(squarePosFlat(squareC),matCO);
  const squareR=makeSegs(squarePosFlat(squareC),matCT);
  root.add(squareW);root.add(squareR);
  s2Data={svd,pts2d,tIM,squareC,squareR,squareW,A};
  initTrails();updateScenario2();updatePanel();updateHUD();updateWristHUD();
}

function updateScenario2(){
  if(!s2Data)return[];
  const tPts=pathScen2(s2Data.pts2d,tParam,s2Data.svd);
  updateIM(s2Data.tIM,tPts);
  const tSquare=pathScen2(s2Data.squareC,tParam,s2Data.svd);
  setLineSegs(s2Data.squareR,squarePos3D(tSquare));
  s2Data.squareW.visible=tParam<0.01;
  return tPts;
}

// ─── Scenario 3: 3D PCA ───────────────────────────────────────────────────────

function buildScenario3(speak, keepPts=false){
  pcaGrabIdx=-1;
  sceneCommon(speak,SCENARIO_NAMES[3]);
  const pts=(keepPts&&s3Data)?s3Data.pts:genPancake3D(60,11);
  const pca=pca3(pts);
  const V=pca.V,mn=pca.mean;
  // Best-fit PC1-PC2 plane (normal = V[:,2])
  const planeMesh=makeSemiPlane([V[0][2],V[1][2],V[2][2]],0x22ff88,0.15,5.5);
  root.add(planeMesh);
  // PC arrows at mean, colored differently
  const pcColors=[0xff4444,0x44ff88,0x4488ff];
  for(let i=0;i<3;i++){
    const d=[V[0][i],V[1][i],V[2][i]],len=Math.max(0.3,pca.s[i]*1.3);
    const mnV=new THREE.Vector3(mn[0],mn[1],mn[2]);
    root.add(new THREE.ArrowHelper(new THREE.Vector3(...d).normalize(),mnV,len,pcColors[i],len*0.22,len*0.1));
    const lbl=makeLabel(`PC${i+1}  σ=${pca.s[i].toFixed(2)}`,SV_COLORS[i]);
    lbl.position.set(mn[0]+d[0]*len*1.4,mn[1]+d[1]*len*1.4,mn[2]+d[2]*len*1.4);root.add(lbl);
  }
  const oIM=makePointCloud(pts,0x00eeff,0.45);
  const tIM=makePointCloud(pts,0xff7722,1.0);
  root.add(oIM);root.add(tIM);
  // White bounding cube (static reference)
  root.add(makeSegs(cubePosArray(buildCubeCorners(pts)),matCO));
  const lbl=makeLabel('3D PCA','#88aaff','big');lbl.position.set(0,2.5,0);root.add(lbl);
  s3Data={pca,pts,tIM,oIM};
  initTrails();updateScenario3();updatePanel();updateHUD();updateWristHUD();
}

function updateScenario3(){
  if(!s3Data)return[];
  const{pca,tIM}=s3Data;
  const rel=pathScen3(pca.centered,tParam,pca);
  const tPts=rel.map(p=>[p[0]+pca.mean[0],p[1]+pca.mean[1],p[2]+pca.mean[2]]);
  updateIM(tIM,tPts);return tPts;
}

// ─── Scenario 4: Least Squares (staged) ──────────────────────────────────────

const S4_PLANES=[[1,1,1,-3],[1,-1,0,0],[0,1,-1,1],[1,0,-2,2],[2,1,0,-4]];
const S4_PCOLORS=[0xff4444,0x44ff44,0x4488ff,0xffaa22,0xff44ff];
const S4_CSS_COLORS=['#ff4444','#44ff44','#4488ff','#ffaa22','#ff44ff'];
let s4PlanesCustom=S4_PLANES.map(p=>[...p]);

// ─── Mode 4 equation input panel (desktop) ────────────────────────────────────
let s4InputPanel=null;
function buildS4InputPanel(){
  s4InputPanel=document.createElement('div');
  s4InputPanel.style.cssText='position:absolute;bottom:12px;right:12px;background:rgba(8,10,28,0.94);'
    +'border:1px solid rgba(80,120,255,0.45);border-radius:10px;padding:14px 16px;'
    +'color:#ccd6f6;font-family:monospace;font-size:13px;display:none;z-index:10;';
  const ttl=document.createElement('div');
  ttl.style.cssText='color:#7799ff;font-weight:bold;margin-bottom:8px;font-size:14px;';
  ttl.textContent='Plane Equations  (Ax + By + Cz + D = 0)';
  s4InputPanel.appendChild(ttl);
  const hdr=document.createElement('div');
  hdr.style.cssText='display:grid;grid-template-columns:36px repeat(4,58px);gap:4px;margin-bottom:4px;color:#5577aa;text-align:center;';
  hdr.innerHTML='<span></span><span>A</span><span>B</span><span>C</span><span>D</span>';
  s4InputPanel.appendChild(hdr);
  for(let i=0;i<5;i++){
    const row=document.createElement('div');
    row.style.cssText='display:grid;grid-template-columns:36px repeat(4,58px);gap:4px;margin-bottom:4px;align-items:center;';
    const lbl=document.createElement('span');
    lbl.style.cssText=`color:${S4_CSS_COLORS[i]};font-weight:bold;`;
    lbl.textContent=`P${i+1}`;
    row.appendChild(lbl);
    for(let j=0;j<4;j++){
      const inp=document.createElement('input');
      inp.type='number';inp.step='0.5';inp.value=s4PlanesCustom[i][j];
      inp.style.cssText='width:100%;background:rgba(20,30,60,0.8);border:1px solid rgba(80,120,255,0.3);'
        +'border-radius:4px;color:#ccd6f6;font-family:monospace;font-size:13px;'
        +'padding:3px 5px;box-sizing:border-box;text-align:center;';
      const ii=i,jj=j;
      inp.addEventListener('change',()=>{
        const v=parseFloat(inp.value);
        if(!isNaN(v)){s4PlanesCustom[ii][jj]=v;if(scenarioMode===4)buildScenario4(false);}
      });
      row.appendChild(inp);
    }
    s4InputPanel.appendChild(row);
  }
  document.body.appendChild(s4InputPanel);
}

function buildScenario4(speak){
  const prevCount=s4Data?s4Data.planeCount:3;
  sceneCommon(speak,SCENARIO_NAMES[4]);
  const planes=s4PlanesCustom;
  const lse3=lseSolve(planes.slice(0,3));
  const lse4=lseSolve(planes.slice(0,4));
  const lse5=lseSolve(planes.slice(0,5));

  // Center each plane at foot of perpendicular from the 5-plane LS solution
  // so the residual endpoint always falls near the plane center
  const planeMeshes=[];
  const PSIZE=7,h=3.5; // larger planes so the dot is clearly visible on them
  for(let i=0;i<5;i++){
    const[a,bv,c,d]=planes[i],nm=Math.sqrt(a*a+bv*bv+c*c);
    const n=[a/nm,bv/nm,c/nm],bi=-d/nm;
    // foot of perpendicular from lse5 solution onto this plane
    const x5=lse5.xLS;
    const dist5=n[0]*x5[0]+n[1]*x5[1]+n[2]*x5[2]-bi;
    const foot=[x5[0]-dist5*n[0],x5[1]-dist5*n[1],x5[2]-dist5*n[2]];
    const pm=makeSemiPlane([n[0],n[1],n[2]],S4_PCOLORS[i],0.22,PSIZE);
    pm.position.set(...foot);pm.material.transparent=true;
    root.add(pm);
    const eq=new THREE.BufferGeometry().setFromPoints(
      [[-h,-h,0],[h,-h,0],[h,h,0],[-h,h,0],[-h,-h,0]].map(v=>new THREE.Vector3(...v)));
    const el=new THREE.Line(eq,new THREE.LineBasicMaterial({color:S4_PCOLORS[i],transparent:true,opacity:0.80}));
    el.position.copy(pm.position);el.quaternion.copy(pm.quaternion);root.add(el);
    if(i>=3){pm.visible=false;el.visible=false;}
    planeMeshes.push({pm,el});
  }

  // Per-plane colored residual lines (one Line per plane)
  const resLines=[];
  for(let i=0;i<5;i++){
    const arr=new Float32Array(6);
    const attr=new THREE.Float32BufferAttribute(arr,3);
    attr.usage=THREE.DynamicDrawUsage;
    const geo=new THREE.BufferGeometry();geo.setAttribute('position',attr);
    const line=new THREE.Line(geo,new THREE.LineBasicMaterial({color:S4_PCOLORS[i],transparent:true,opacity:0.95,linewidth:2}));
    line.visible=false;root.add(line);
    resLines.push({line,attr,arr});
  }

  // Pairwise intersection lines — visible only when planeCount===3
  const IPAIRS=[[0,1],[0,2],[1,2]];
  const intersectLines=[];
  for(const[i,j] of IPAIRS){
    const seg=computeIntersectLine(planes[i],planes[j],5);
    if(seg){
      const geo=new THREE.BufferGeometry().setFromPoints(
        [new THREE.Vector3(...seg.start),new THREE.Vector3(...seg.end)]);
      const col=new THREE.Color(S4_PCOLORS[i]).lerp(new THREE.Color(S4_PCOLORS[j]),0.5);
      const l=new THREE.Line(geo,new THREE.LineBasicMaterial({color:col,transparent:true,opacity:0.90}));
      l.visible=false;root.add(l);
      intersectLines.push(l);
    }else{intersectLines.push(null);}
  }

  // LS sphere — smaller, high emissive for bloom
  const lsMat=new THREE.MeshStandardMaterial({color:0xffffff,emissive:0xffffff,emissiveIntensity:1.5});
  const lsSphere=new THREE.Mesh(new THREE.SphereGeometry(0.04,16,12),lsMat);
  lsSphere.position.set(...lse3.xLS);root.add(lsSphere);
  // Glow + trail
  root.add(s4GlowSprite);s4GlowSprite.visible=true;
  root.add(s4TrailLine);s4TrailData=[];s4TrailFadeAge=-1;s4TrailMat.opacity=0;s4TrailLine.visible=false;

  // Bounding cube
  const allXLS=[lse3.xLS,lse4.xLS,lse5.xLS];
  const pad=2.5;
  const mn=[Math.min(...allXLS.map(p=>p[0]))-pad,Math.min(...allXLS.map(p=>p[1]))-pad,Math.min(...allXLS.map(p=>p[2]))-pad];
  const mx=[Math.max(...allXLS.map(p=>p[0]))+pad,Math.max(...allXLS.map(p=>p[1]))+pad,Math.max(...allXLS.map(p=>p[2]))+pad];
  root.add(makeSegs(cubePosArray(buildCubeCorners([mn,mx])),matCO));

  const lbl=makeLabel('Least Squares','#88aaff','big');lbl.position.set(0,2.65,0);root.add(lbl);

  s4Data={lse3,lse4,lse5,planes,planeMeshes,resLines,intersectLines,lsSphere,lsMat,
    planeCount:prevCount,animFrom:null,animTo:null,animProgress:1};
  initTrails();updateScenario4();updatePanel();updateHUD();updateWristHUD();
}

function updateScenario4(){
  if(!s4Data)return[];
  const{lse3,lse4,lse5,planes,planeMeshes,resLines,lsSphere,lsMat}=s4Data;
  const cnt=s4Data.planeCount;

  // Sphere position: smooth-step interpolation during animation
  const ss=x=>x*x*(3-2*x);
  const ap=ss(Math.min(1,s4Data.animProgress));
  const lsPos=s4Data.animFrom&&ap<1
    ? s4Data.animFrom.map((v,i)=>v+(s4Data.animTo[i]-v)*ap)
    : [lse3,lse4,lse5][cnt-3].xLS.slice();
  lsSphere.position.set(...lsPos);

  // Bloom pulse during animation
  const pulse=s4Data.animProgress<1?Math.sin(Math.PI*ap):0;
  lsMat.emissiveIntensity=1.5+4.0*pulse;
  lsSphere.scale.setScalar(1+0.25*pulse);
  // Glow sprite — follows sphere, brightens during animation
  s4GlowSprite.position.copy(lsSphere.position);
  s4GlowMat.opacity=0.45+0.55*pulse;

  // Plane visibility — full opacity immediately
  for(let i=0;i<3;i++){planeMeshes[i].pm.visible=true;planeMeshes[i].el.visible=true;}
  planeMeshes[3].pm.visible=cnt>=4;planeMeshes[3].el.visible=cnt>=4;
  if(cnt>=4){planeMeshes[3].pm.material.opacity=0.22;planeMeshes[3].el.material.opacity=0.80;}
  planeMeshes[4].pm.visible=cnt>=5;planeMeshes[4].el.visible=cnt>=5;
  if(cnt>=5){planeMeshes[4].pm.material.opacity=0.22;planeMeshes[4].el.material.opacity=0.80;}

  // Intersection lines — only when 3 planes active
  if(s4Data.intersectLines){
    for(const l of s4Data.intersectLines){if(l)l.visible=(cnt===3);}
  }

  // Per-plane colored residual lines from sphere to foot on each plane
  for(let i=0;i<5;i++){
    const{line,attr,arr}=resLines[i];
    if(i<cnt){
      const[a,bv,c,d]=planes[i],nm=Math.sqrt(a*a+bv*bv+c*c);
      const n=[a/nm,bv/nm,c/nm],bi=-d/nm;
      const dist=n[0]*lsPos[0]+n[1]*lsPos[1]+n[2]*lsPos[2]-bi;
      arr[0]=lsPos[0];arr[1]=lsPos[1];arr[2]=lsPos[2];
      arr[3]=lsPos[0]-dist*n[0];arr[4]=lsPos[1]-dist*n[1];arr[5]=lsPos[2]-dist*n[2];
      attr.array.set(arr);attr.needsUpdate=true;
      line.visible=true;
    } else {
      line.visible=false;
    }
  }
  return[];
}

// ─── Controllers ─────────────────────────────────────────────────────────────

const ctrl1=renderer.xr.getController(0);
const ctrl2=renderer.xr.getController(1);
scene.add(ctrl1);scene.add(ctrl2);

// Map handedness→ctrlGrp so grab/wristHUD stay on correct hand after headset reconnect
const ctrlByHand={};
function _bindCtrl(c){
  c.addEventListener('connected',e=>{if(e.data.handedness==='right'||e.data.handedness==='left')ctrlByHand[e.data.handedness]=c;});
  c.addEventListener('disconnected',()=>{for(const k of Object.keys(ctrlByHand)){if(ctrlByHand[k]===c)delete ctrlByHand[k];}});
}
_bindCtrl(ctrl1);_bindCtrl(ctrl2);

function addRay(c,col=0xffffff){
  const g=new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(0,0,0),new THREE.Vector3(0,0,-1)]);
  const l=new THREE.Line(g,new THREE.LineBasicMaterial({color:col,transparent:true,opacity:0.5}));
  l.scale.z=5;c.add(l);
}
addRay(ctrl1,0xffffff);addRay(ctrl2,0x44ffaa);

let prevGripPressed=false;

// ─── Teleportation ────────────────────────────────────────────────────────────

let baseRefSpace=null,teleportTarget=null,prevThumbPressed=false,currentXRSession=null;

// ─── Grab / throw state ───────────────────────────────────────────────────────
const grabCtrlPos=new THREE.Vector3(),grabCtrlQuat=new THREE.Quaternion();
const throwVel=new THREE.Vector3();
let grabActive=false,prevAPressed=false;
let pcaGrabIdx=-1;
// Pre-allocated temps (avoid per-frame allocation)
const _cPos=new THREE.Vector3(),_cQuat=new THREE.Quaternion();
const _dPos=new THREE.Vector3(),_dQuat=new THREE.Quaternion(),_invQ=new THREE.Quaternion(),_rp=new THREE.Vector3();
const _invRootMat=new THREE.Matrix4(),_localPos=new THREE.Vector3();
renderer.xr.addEventListener('sessionstart',()=>{baseRefSpace=renderer.xr.getReferenceSpace();currentXRSession=renderer.xr.getSession();wristHUDAttached=false;});
renderer.xr.addEventListener('sessionend',()=>{baseRefSpace=null;currentXRSession=null;wristHUDAttached=false;grabActive=false;prevAPressed=false;vrGrabMode=false;grabbedPlaneIdx=-1;});

function triggerHaptics(intensity=0.65,duration=180){
  if(!currentXRSession)return;
  for(const src of currentXRSession.inputSources){
    const gp=src.gamepad;if(!gp)continue;
    // Try both haptic APIs — Quest may expose either or both
    if(gp.hapticActuators?.length>0){
      gp.hapticActuators[0].pulse(intensity,duration);
    } else if(gp.vibrationActuator){
      gp.vibrationActuator.playEffect('dual-rumble',{
        duration,strongMagnitude:intensity,weakMagnitude:intensity*0.5
      }).catch(()=>{});
    }
  }
}

const teleportRay=new THREE.Raycaster();
const grabRay=new THREE.Raycaster();
const tmpMatrix=new THREE.Matrix4();

function doTeleport(pos){
  if(!baseRefSpace||typeof XRRigidTransform==='undefined')return;
  const t=new XRRigidTransform({x:-pos.x,y:0,z:-pos.z,w:1},{x:0,y:0,z:0,w:1});
  renderer.xr.setReferenceSpace(baseRefSpace.getOffsetReferenceSpace(t));
}


// ─── Speech synthesis ─────────────────────────────────────────────────────────

const STAGE_SPEECH=[
  'Stage 1. Rotating by V, aligning to the right singular vectors.',
  'Stage 2. Scaling along the principal axes by the singular values.',
  'Stage 3. Rotating by U, arriving at the final matrix.',
];
let lastSpokenStage=-1,speechEnabled=true;

function speakText(text){
  if(!speechEnabled||!window.speechSynthesis)return;
  window.speechSynthesis.cancel();
  const u=new SpeechSynthesisUtterance(text);
  u.rate=0.88;u.pitch=1.0;u.volume=0.9;
  window.speechSynthesis.speak(u);
}

// ─── B button state (right controller) ────────────────────────────────────────
let prevBPressed=false;

// ─── Mode 4 trigger edge-detection ────────────────────────────────────────────
let s4PrevRightTrig=false,s4PrevLeftTrig=false;

// ─── Keyboard ─────────────────────────────────────────────────────────────────

const keys={};
window.addEventListener('keydown',e=>{
  keys[e.key]=true;
  if(e.key==='1'){presetIdx=0;tParam=0;mat0Custom=deepCopy2D(PRESETS[0].A);rebuildScene(true);}
  if(e.key==='2'){presetIdx=1;tParam=0;mat0Custom=deepCopy2D(PRESETS[1].A);rebuildScene(true);}
  if(e.key==='3'){presetIdx=2;tParam=0;mat0Custom=deepCopy2D(PRESETS[2].A);rebuildScene(true);}
  if(e.key.toLowerCase()==='g'){presetIdx=(presetIdx+1)%PRESETS.length;tParam=0;mat0Custom=deepCopy2D(PRESETS[presetIdx].A);rebuildScene(true);}
  if(e.key==='='||e.key==='+'){rootScale=Math.min(2.0,rootScale+0.1);root.scale.setScalar(rootScale);updateHUD();}
  if(e.key==='-'){rootScale=Math.max(0.1,rootScale-0.1);root.scale.setScalar(rootScale);updateHUD();}
  if(e.key.toLowerCase()==='s'){scenarioMode=(scenarioMode+1)%5;planeEditMode=false;matrixEditMode=false;mat0Custom=deepCopy2D(PRESETS[presetIdx].A);mat1Custom=deepCopy2D(MAT1_DEFAULT);mat2Custom=deepCopy2D(MAT2_DEFAULT);tParam=0;rebuildScene(true);}
  // Mode 4 discrete plane add/remove via arrow keys
  if(e.key==='ArrowRight'&&scenarioMode===4&&s4Data&&s4Data.planeCount<5){
    s4Data.animFrom=s4Data.lsSphere.position.toArray();
    s4Data.planeCount++;
    s4Data.animTo=[s4Data.lse3,s4Data.lse4,s4Data.lse5][s4Data.planeCount-3].xLS.slice();
    s4Data.animProgress=0;
  }
  if(e.key==='ArrowLeft'&&scenarioMode===4&&s4Data&&s4Data.planeCount>3){
    s4Data.animFrom=s4Data.lsSphere.position.toArray();
    s4Data.planeCount--;
    s4Data.animTo=[s4Data.lse3,s4Data.lse4,s4Data.lse5][s4Data.planeCount-3].xLS.slice();
    s4Data.animProgress=0;
  }
});
window.addEventListener('keyup',e=>{keys[e.key]=false;});
window.addEventListener('resize',()=>{
  camera.aspect=window.innerWidth/window.innerHeight;camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth,window.innerHeight);
  composer.setSize(window.innerWidth,window.innerHeight);
  bloomPass.setSize(window.innerWidth,window.innerHeight);
});

// ─── Init ─────────────────────────────────────────────────────────────────────

buildS4InputPanel();
rebuildScene();updateHUD();updateWristHUD();

// ─── Render loop ──────────────────────────────────────────────────────────────

const clock=new THREE.Clock();
const T_SPEED=0.7;

renderer.setAnimationLoop(()=>{
  const dt=Math.min(clock.getDelta(),0.05);
  let moved=false;

  // Desktop t scrub (skipped in mode 4 — arrow keys handled via keydown instead)
  if(keys['ArrowRight']&&scenarioMode!==4){tParam=Math.min(3,tParam+T_SPEED*dt);moved=true;}
  if(keys['ArrowLeft'] &&scenarioMode!==4){tParam=Math.max(0,tParam-T_SPEED*dt);moved=true;}

  // VR input
  if(renderer.xr.isPresenting){
    const session=renderer.xr.getSession();
    let leftCtrl=null;
    if(session){
      for(let i=0;i<session.inputSources.length;i++){
        const src=session.inputSources[i];
        if(!src.gamepad)continue;
        const ctrlGrp=ctrlByHand[src.handedness]??(i===0?ctrl1:ctrl2);
        const trigger=src.gamepad.buttons[0]?.pressed??false;
        const grip   =src.gamepad.buttons[1]?.pressed??false;
        const thumbBtn=src.gamepad.buttons[3]?.pressed??false;

        if(src.handedness==='right'){
          if(vrGrabMode&&scenarioMode===4&&s4Data){
            // ── Plane grab mode ────────────────────────────────────────────────
            ctrlGrp.getWorldPosition(_cPos);
            _cQuat.setFromRotationMatrix(ctrlGrp.matrixWorld);
            // Hover: ray cast from controller forward direction → pick plane
            grabRay.ray.origin.copy(_cPos);
            tmpMatrix.identity().extractRotation(ctrlGrp.matrixWorld);
            grabRay.ray.direction.set(0,0,-1).applyMatrix4(tmpMatrix);
            const visMeshes=s4Data.planeMeshes.filter(p=>p.pm.visible).map(p=>p.pm);
            const rayHits=grabRay.intersectObjects(visMeshes);
            hoverPlaneIdx=rayHits.length>0
              ?s4Data.planeMeshes.findIndex(p=>p.pm===rayHits[0].object):-1;
            // Visual highlight
            for(let pi=0;pi<5;pi++){
              if(!s4Data.planeMeshes[pi].pm.visible)continue;
              const hot=pi===hoverPlaneIdx||pi===grabbedPlaneIdx;
              s4Data.planeMeshes[pi].pm.material.opacity=hot?0.55:0.22;
              s4Data.planeMeshes[pi].el.material.opacity=hot?1.0:0.55;
            }
            // Grip → grab / release
            if(grip&&!prevPlaneGrip&&hoverPlaneIdx>=0&&grabbedPlaneIdx<0){
              grabbedPlaneIdx=hoverPlaneIdx;
              planeGrabPrevPos.copy(_cPos);
              planeGrabPrevQuat.copy(_cQuat);
              triggerHaptics(0.6,120);
            } else if(!grip&&grabbedPlaneIdx>=0){
              grabbedPlaneIdx=-1;
            }
            // Drag grabbed plane (translate + rotate)
            if(grip&&grabbedPlaneIdx>=0){
              const prevRL=root.worldToLocal(planeGrabPrevPos.clone());
              const currRL=root.worldToLocal(_cPos.clone());
              _dPos.copy(currRL).sub(prevRL);
              const pm=s4Data.planeMeshes[grabbedPlaneIdx].pm;
              const el=s4Data.planeMeshes[grabbedPlaneIdx].el;
              pm.position.add(_dPos);
              // Delta rotation: world → root-local (dQL = rootQ^-1 * dQWorld * rootQ)
              _invQ.copy(planeGrabPrevQuat).invert();
              _dQuat.copy(_cQuat).multiply(_invQ);
              const dQL=root.quaternion.clone().invert();
              dQL.multiply(_dQuat).multiply(root.quaternion);
              _rp.copy(pm.position).sub(currRL);
              _rp.applyQuaternion(dQL);
              pm.position.copy(currRL).add(_rp);
              pm.quaternion.premultiply(dQL);
              el.position.copy(pm.position);el.quaternion.copy(pm.quaternion);
              planeGrabPrevPos.copy(_cPos);planeGrabPrevQuat.copy(_cQuat);
              moved=true;
            }
            // Hand open/close animation
            const tgt=grabbedPlaneIdx>=0?1.0:hoverPlaneIdx>=0?0.45:0.0;
            vrHandAnimT=THREE.MathUtils.lerp(vrHandAnimT,tgt,Math.min(1,dt*9));
            updateVRHandCurl(vrHand.fingerChains,vrHand.thumbChain,vrHandAnimT);
            prevPlaneGrip=grip;
          } else {
            // ── Normal right-controller inputs ─────────────────────────────────
            prevPlaneGrip=false;
            if(scenarioMode===4){
              if(trigger&&!s4PrevRightTrig&&s4Data&&s4Data.planeCount<5){
                s4Data.animFrom=s4Data.lsSphere.position.toArray();
                s4Data.planeCount++;
                s4Data.animTo=[s4Data.lse3,s4Data.lse4,s4Data.lse5][s4Data.planeCount-3].xLS.slice();
                s4Data.animProgress=0;triggerHaptics(0.5,120);moved=true;
              }
              s4PrevRightTrig=trigger;
            } else {
              if(trigger){tParam=Math.min(3,tParam+T_SPEED*dt);moved=true;}
            }
          }
          if(grip&&!prevGripPressed&&!vrGrabMode){presetIdx=(presetIdx+1)%PRESETS.length;tParam=0;mat0Custom=deepCopy2D(PRESETS[presetIdx].A);rebuildScene(true);}
          prevGripPressed=grip;
          // Right thumbstick Y → zoom
          const stickY=src.gamepad.axes[3]??src.gamepad.axes[1]??0;
          if(Math.abs(stickY)>0.15){
            rootScale=THREE.MathUtils.clamp(rootScale-stickY*0.9*dt,0.10,2.0);
            root.scale.setScalar(rootScale);moved=true;
          }
          // A button (buttons[4]) → mode 3: grab PCA point | else: grab/rotate/throw scene
          const aBtn=src.gamepad.buttons[4]?.pressed??false;
          if(scenarioMode===3&&s3Data){
            _cPos.setFromMatrixPosition(ctrlGrp.matrixWorld);
            if(aBtn&&!prevAPressed){
              // Find nearest point in root-local space within threshold 0.22
              _invRootMat.copy(root.matrixWorld).invert();
              _localPos.copy(_cPos).applyMatrix4(_invRootMat);
              let best=-1,bestD=0.22*0.22;
              for(let i=0;i<s3Data.pts.length;i++){
                const p=s3Data.pts[i];
                const dx=_localPos.x-p[0],dy=_localPos.y-p[1],dz=_localPos.z-p[2];
                const d2=dx*dx+dy*dy+dz*dz;
                if(d2<bestD){bestD=d2;best=i;}
              }
              if(best>=0){pcaGrabIdx=best;triggerHaptics(0.6,150);}
            } else if(aBtn&&pcaGrabIdx>=0){
              // Drag: update grabbed point to controller's root-local position
              _invRootMat.copy(root.matrixWorld).invert();
              _localPos.copy(_cPos).applyMatrix4(_invRootMat);
              s3Data.pts[pcaGrabIdx][0]=_localPos.x;
              s3Data.pts[pcaGrabIdx][1]=_localPos.y;
              s3Data.pts[pcaGrabIdx][2]=_localPos.z;
              dummy.position.copy(_localPos);dummy.updateMatrix();
              s3Data.oIM.setMatrixAt(pcaGrabIdx,dummy.matrix);
              s3Data.oIM.instanceMatrix.needsUpdate=true;
            } else if(!aBtn&&prevAPressed&&pcaGrabIdx>=0){
              // Release: rebuild PCA keeping moved points
              buildScenario3(false,true);
            } else if(!aBtn){
              pcaGrabIdx=-1;
            }
          } else {
            if(aBtn){
              _cPos.setFromMatrixPosition(ctrlGrp.matrixWorld);
              _cQuat.setFromRotationMatrix(ctrlGrp.matrixWorld);
              if(!prevAPressed){
                // Grab start — record controller pose, reset throw velocity
                grabCtrlPos.copy(_cPos);grabCtrlQuat.copy(_cQuat);
                throwVel.set(0,0,0);grabActive=true;
              } else if(grabActive){
                // Delta position: move root with the hand
                _dPos.copy(_cPos).sub(grabCtrlPos);
                throwVel.copy(_dPos).divideScalar(Math.max(dt,0.001));
                root.position.add(_dPos);
                // Delta rotation: rotate root around the controller (grab point)
                _invQ.copy(grabCtrlQuat).invert();
                _dQuat.copy(_cQuat).multiply(_invQ);
                _rp.copy(root.position).sub(_cPos).applyQuaternion(_dQuat);
                root.position.copy(_cPos).add(_rp);
                root.quaternion.premultiply(_dQuat);
                grabCtrlPos.copy(_cPos);grabCtrlQuat.copy(_cQuat);
              }
            } else if(prevAPressed){
              grabActive=false; // release — throwVel carries the momentum
            }
          }
          prevAPressed=aBtn;
          // B button → toggle matrix editor (modes 0-2) or plane editor (mode 4)
          const bBtn=src.gamepad.buttons[5]?.pressed??false;
          if(bBtn&&!prevBPressed){
            if(scenarioMode===4){planeEditMode=!planeEditMode;triggerHaptics(0.4,100);updatePanel();updateHUD();updateWristHUD();}
            else if(scenarioMode<3){matrixEditMode=!matrixEditMode;if(matrixEditMode){matEditRow=0;matEditCol=0;}triggerHaptics(0.4,100);updatePanel();updateHUD();updateWristHUD();}
          }
          prevBPressed=bBtn;
        }
        if(src.handedness==='left'){
          leftCtrl=ctrlGrp;
          if(planeEditMode&&scenarioMode===4){
            // ── Plane editor controls ──────────────────────────────────────────
            const lsX=src.gamepad.axes[2]??src.gamepad.axes[0]??0;
            const lsY=src.gamepad.axes[3]??src.gamepad.axes[1]??0;
            // Left/right → select column (A B C D)
            if(lsX>0.5&&!editPrevStickX){
              planeEditCol=(planeEditCol+1)%4;
              editPrevStickX=true;triggerHaptics(0.25,50);
              updatePanel();updateWristHUD();
            } else if(lsX<-0.5&&!editPrevStickX){
              planeEditCol=(planeEditCol+3)%4;
              editPrevStickX=true;triggerHaptics(0.25,50);
              updatePanel();updateWristHUD();
            } else if(Math.abs(lsX)<0.3) editPrevStickX=false;
            // Up/down → ±0.5 on selected coefficient
            if(lsY<-0.5&&!editPrevStickY){
              s4PlanesCustom[planeEditRow][planeEditCol]+=0.5;
              editPrevStickY=true;triggerHaptics(0.4,80);rebuildScene(false);
            } else if(lsY>0.5&&!editPrevStickY){
              s4PlanesCustom[planeEditRow][planeEditCol]-=0.5;
              editPrevStickY=true;triggerHaptics(0.4,80);rebuildScene(false);
            } else if(Math.abs(lsY)<0.3) editPrevStickY=false;
            // Left grip → cycle to next plane
            if(grip&&!editPrevLGrip){
              planeEditRow=(planeEditRow+1)%5;
              editPrevLGrip=true;triggerHaptics(0.3,60);
              updatePanel();updateWristHUD();
            } else if(!grip) editPrevLGrip=false;
          } else if(matrixEditMode&&scenarioMode<3){
            // ── Matrix editor controls ─────────────────────────────────────────
            const lsX=src.gamepad.axes[2]??src.gamepad.axes[0]??0;
            const lsY=src.gamepad.axes[3]??src.gamepad.axes[1]??0;
            const matRows=scenarioMode===1?2:3,matCols=scenarioMode===2?2:3;
            const mat=scenarioMode===0?mat0Custom:scenarioMode===1?mat1Custom:mat2Custom;
            // Left/right → select column
            if(lsX>0.5&&!matEditPrevStickX){
              matEditCol=(matEditCol+1)%matCols;matEditPrevStickX=true;triggerHaptics(0.25,50);updatePanel();updateWristHUD();
            } else if(lsX<-0.5&&!matEditPrevStickX){
              matEditCol=(matEditCol+matCols-1)%matCols;matEditPrevStickX=true;triggerHaptics(0.25,50);updatePanel();updateWristHUD();
            } else if(Math.abs(lsX)<0.3) matEditPrevStickX=false;
            // Up/down → ±0.1 on selected element
            if(lsY<-0.5&&!matEditPrevStickY){
              mat[matEditRow][matEditCol]=Math.round((mat[matEditRow][matEditCol]+0.1)*100)/100;
              matEditPrevStickY=true;triggerHaptics(0.4,80);rebuildScene(false);
            } else if(lsY>0.5&&!matEditPrevStickY){
              mat[matEditRow][matEditCol]=Math.round((mat[matEditRow][matEditCol]-0.1)*100)/100;
              matEditPrevStickY=true;triggerHaptics(0.4,80);rebuildScene(false);
            } else if(Math.abs(lsY)<0.3) matEditPrevStickY=false;
            // Left grip → cycle to next row
            if(grip&&!matEditPrevLGrip){
              matEditRow=(matEditRow+1)%matRows;matEditPrevLGrip=true;triggerHaptics(0.3,60);updatePanel();updateWristHUD();
            } else if(!grip) matEditPrevLGrip=false;
          } else {
            // ── Normal controls ────────────────────────────────────────────────
            if(scenarioMode===4){
              if(trigger&&!s4PrevLeftTrig&&s4Data&&s4Data.planeCount>3){
                s4Data.animFrom=s4Data.lsSphere.position.toArray();
                s4Data.planeCount--;
                s4Data.animTo=[s4Data.lse3,s4Data.lse4,s4Data.lse5][s4Data.planeCount-3].xLS.slice();
                s4Data.animProgress=0;triggerHaptics(0.5,120);moved=true;
              }
              s4PrevLeftTrig=trigger;
            } else {
              if(trigger){tParam=Math.max(0,tParam-T_SPEED*dt);moved=true;}
            }
            if(thumbBtn&&!prevThumbPressed&&teleportTarget)doTeleport(teleportTarget);
            prevThumbPressed=thumbBtn;
            // Left thumbstick X → cycle scenario
            const leftStickX=src.gamepad.axes[2]??src.gamepad.axes[0]??0;
            if(leftStickX>0.5&&!prevLeftStickTriggered){
              scenarioMode=(scenarioMode+1)%5;planeEditMode=false;matrixEditMode=false;
              mat0Custom=deepCopy2D(PRESETS[presetIdx].A);mat1Custom=deepCopy2D(MAT1_DEFAULT);mat2Custom=deepCopy2D(MAT2_DEFAULT);
              tParam=0;triggerHaptics(0.5,120);rebuildScene(true);
              prevLeftStickTriggered=true;
            } else if(leftStickX<-0.5&&!prevLeftStickTriggered){
              scenarioMode=(scenarioMode+4)%5;planeEditMode=false;matrixEditMode=false;
              mat0Custom=deepCopy2D(PRESETS[presetIdx].A);mat1Custom=deepCopy2D(MAT1_DEFAULT);mat2Custom=deepCopy2D(MAT2_DEFAULT);
              tParam=0;triggerHaptics(0.5,120);rebuildScene(true);
              prevLeftStickTriggered=true;
            } else if(Math.abs(leftStickX)<0.3){
              prevLeftStickTriggered=false;
            }
          }
          // X button (buttons[4]) → enter / exit plane grab mode (mode 4 only)
          const xBtn=src.gamepad.buttons[4]?.pressed??false;
          if(scenarioMode===4&&!planeEditMode){
            if(xBtn&&!prevXBtn){
              vrGrabMode=true;grabbedPlaneIdx=-1;hoverPlaneIdx=-1;
              triggerHaptics(0.4,100);updateHUD();
            } else if(!xBtn&&prevXBtn&&vrGrabMode){
              vrGrabMode=false;grabbedPlaneIdx=-1;hoverPlaneIdx=-1;
              // Restore plane visuals
              if(s4Data)for(let pi=0;pi<5;pi++){
                s4Data.planeMeshes[pi].pm.material.opacity=0.22;
                s4Data.planeMeshes[pi].el.material.opacity=0.80;
              }
              // Derive new plane equations from mesh transforms and rebuild
              if(s4Data){
                const _n=new THREE.Vector3();
                for(let i=0;i<5;i++){
                  const pm=s4Data.planeMeshes[i].pm;
                  _n.set(0,0,1).applyQuaternion(pm.quaternion);
                  s4PlanesCustom[i]=[_n.x,_n.y,_n.z,-_n.dot(pm.position)];
                }
                rebuildScene(false);
              }
              triggerHaptics(0.6,200);updateHUD();
            }
          } else if(scenarioMode!==4) vrGrabMode=false;
          prevXBtn=xBtn;
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

  // VR hand visibility — show/hide and sync transform with right controller
  const _rightCtrl=ctrlByHand['right'];
  const _shouldShowHand=vrGrabMode&&scenarioMode===4&&renderer.xr.isPresenting&&!!_rightCtrl;
  if(_shouldShowHand!==vrHand.grp.visible){
    vrHand.grp.visible=_shouldShowHand;
    if(_rightCtrl)_rightCtrl.children.forEach(c=>c.visible=!_shouldShowHand);
  }
  if(_shouldShowHand&&_rightCtrl){
    _rightCtrl.getWorldPosition(vrHand.grp.position);
    _cQuat.setFromRotationMatrix(_rightCtrl.matrixWorld);
    vrHand.grp.quaternion.copy(_cQuat).multiply(_vrHandOffset);
  }

  // Mode 4: continuously advance sphere animation independent of triggers
  if(scenarioMode===4&&s4Data&&s4Data.animProgress<1){
    s4Data.animProgress=Math.min(1,s4Data.animProgress+dt*2.5);
    updateScenario4();
    s4AddTrailPoint(s4Data.lsSphere.position);
    if(s4Data.animProgress>=1)s4TrailFadeAge=0; // begin fade when animation finishes
    updatePanel();updateHUD();updateWristHUD();
  }
  // Fade trail out after animation ends
  if(s4TrailFadeAge>=0){
    s4TrailFadeAge+=dt;
    s4TrailMat.opacity=Math.max(0,1-s4TrailFadeAge/1.8);
    if(s4TrailFadeAge>1.8){s4TrailLine.visible=false;s4TrailFadeAge=-1;}
  }
  // Hide glow when not in mode 4
  if(scenarioMode!==4)s4GlowSprite.visible=false;

  // Throw physics — apply velocity with exponential damping
  if(!grabActive&&throwVel.lengthSq()>0.005){
    root.position.addScaledVector(throwVel,dt);
    throwVel.multiplyScalar(Math.max(0,1-6*dt));
    if(throwVel.lengthSq()<0.005)throwVel.set(0,0,0);
  }

  // Handle move: update scene, trails, panel, HUD, pulse
  if(moved){
    const tPts=updateSceneForT();
    pushTrail(tPts);
    updatePanel();updateHUD();updateWristHUD();
    // Stage change pulse + speech
    const newFloor=Math.min(2,Math.floor(tParam));
    if(newFloor!==lastTFloor&&tParam>0.02){triggerPulse();triggerHaptics();lastTFloor=newFloor;}
    if(newFloor!==lastSpokenStage&&tParam>0.05){speakText(STAGE_SPEECH[newFloor]);lastSpokenStage=newFloor;}
  }

  // Animate trails (fade out)
  for(let i=0;i<TRAIL_N;i++){
    if(trailAges[i]>=0){
      trailAges[i]+=dt;
      const op=TRAIL_MAX_OP*Math.exp(-trailAges[i]*TRAIL_DECAY);
      trailPts[i].material.opacity=op;
      if(op<0.004){trailPts[i].visible=false;trailAges[i]=-1;}
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
