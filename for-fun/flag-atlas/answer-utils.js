(function(root){
'use strict';
function normalize(value){return String(value).normalize('NFD').replace(/[\u0300-\u036f]/g,'').toLowerCase().replace(/&/g,' and ').replace(/\bsaint\b/g,'st').replace(/[^a-z0-9]/g,'').replace(/^the/,'');}
// Optimal-string-alignment distance treats a swapped adjacent pair as one typo.
function distance(a,b){const d=Array.from({length:a.length+1},()=>Array(b.length+1).fill(0));for(let i=0;i<=a.length;i++)d[i][0]=i;for(let j=0;j<=b.length;j++)d[0][j]=j;for(let i=1;i<=a.length;i++)for(let j=1;j<=b.length;j++){d[i][j]=Math.min(d[i-1][j]+1,d[i][j-1]+1,d[i-1][j-1]+(a[i-1]===b[j-1]?0:1));if(i>1&&j>1&&a[i-1]===b[j-2]&&a[i-2]===b[j-1])d[i][j]=Math.min(d[i][j],d[i-2][j-2]+1)}return d[a.length][b.length]}
function tolerance(length){return length<5?0:length<9?1:length<14?2:3;}
function names(country){return [...new Set([country.name,...(country.aliases||[])].map(normalize).filter(Boolean))]}
function evaluate(country,answer,countries){const value=normalize(answer);if(!value)return {correct:false,kind:'empty'};if(['congo','korea','virginislands'].includes(value))return {correct:false,kind:'ambiguous'};const pool=countries&&countries.length?countries:[country];const exact=pool.filter(c=>names(c).includes(value));if(exact.length)return {correct:exact.length===1&&exact[0].code===country.code,kind:exact.length>1?'ambiguous':'exact'};
// Do not accept an incomplete geographic qualifier, or fuzz abbreviations.
if(['north','south','east','west','republic','democraticrepublic','guinea','newguinea','saint','st'].includes(value))return {correct:false,kind:'ambiguous'};
let best=Infinity,winners=[];for(const c of pool){let score=Infinity;for(const alias of names(c)){const limit=tolerance(Math.min(alias.length,value.length));if(!limit||Math.abs(alias.length-value.length)>limit)continue;const delta=distance(value,alias);if(delta<=limit)score=Math.min(score,delta)}if(score<best){best=score;winners=[c.code]}else if(Number.isFinite(score)&&score===best)winners.push(c.code)}return {correct:Number.isFinite(best)&&winners.length===1&&winners[0]===country.code,kind:winners.length>1?'ambiguous':'typo'};}
function matches(country,answer,countries){return evaluate(country,answer,countries).correct}
const api={normalize,distance,tolerance,evaluate,matches};if(typeof module!=='undefined'&&module.exports)module.exports=api;else root.FlagAnswers=api;
})(typeof window!=='undefined'?window:globalThis);
