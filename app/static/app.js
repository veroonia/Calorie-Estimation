const imageInput = document.getElementById("imageInput");
const preview = document.getElementById("preview");
const result = document.getElementById("result");
const loader = document.getElementById("loader");

imageInput.onchange = () => {
    const file = imageInput.files[0];
    if (!file) return;

    preview.src = URL.createObjectURL(file);
    preview.hidden = false;
};

async function submitImage() {
    const file = imageInput.files[0];
    if (!file) return alert("Please select an image");

    result.hidden = true;
    loader.hidden = false;

    const formData = new FormData();
    formData.append("file", file);

    const response = await fetch("/predict-image", {
        method: "POST",
        body: formData
    });

    loader.hidden = true;

    if (!response.ok) {
        alert("Prediction failed");
        return;
    }

    const data = await response.json();
    document.getElementById("food").innerText = data.food_category;
    document.getElementById("calories").innerText = data.calories_per_100g;
    result.hidden = false;
}
