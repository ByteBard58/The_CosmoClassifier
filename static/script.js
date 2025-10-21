const form = document.getElementById('predict-form')
const result = document.getElementById('result')
const predText = document.getElementById('pred-text')

form.addEventListener('submit', async (e)=>{
  e.preventDefault()
  const fd = new FormData(form)
  const payload = {}
  for (const [k,v] of fd.entries()) payload[k]=v
  // send JSON
  const res = await fetch('/predict',{
    method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)
  })
  if (!res.ok){
    const err = await res.json().catch(()=>({error:'server error'}))
    alert(err.error||'Prediction failed')
    return
  }
  const data = await res.json()
  predText.innerText = `Predicted: ${data.prediction}`
  // update progress bars
  for (const k of Object.keys(data.probs)){
    const val = data.probs[k]
    const bar = document.getElementById('p-'+k)
    const txt = document.getElementById('v-'+k)
    if(bar) bar.value = val
    if(txt) txt.innerText = `${(val*100).toFixed(1)}%`
  }
  result.classList.remove('hidden')
})