// To connect the real backend, replace the mock below with:
// const response = await fetch('http://localhost:8000/predict', {
//   method: 'POST',
//   body: formData   <-- FormData with file appended
// })
// return await response.json()

function fileToBase64(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onload = () => {
      // Strip the data URL prefix (e.g. "data:image/png;base64,") to return raw base64
      const base64 = reader.result.split(',')[1]
      resolve(base64)
    }
    reader.onerror = () => reject(reader.error)
    reader.readAsDataURL(file)
  })
}

export async function predict(file) {
  const base64 = await fileToBase64(file)

  await new Promise((r) => setTimeout(r, 2000))

  return {
    overlay:         base64,
    dice_score:      0.847,
    confidence:      91.3,
    processing_time: 1.2,
  }
}
