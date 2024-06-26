document.getElementById('predictionForm').addEventListener('submit', function(event) {
    event.preventDefault();
    const date = document.getElementById('date').value;
    const model = document.getElementById('model').value;

    fetch('/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ date: date, model: model })
    })
    .then(response => response.json())
    .then(data => {
        const result = `Kết quả dự đoán cho ngày ${data.date} là ${data.predicted_price}.`;
        document.getElementById('result').innerText = result;

        // Cập nhật ảnh theo mô hình
        const imageElement = document.getElementById('modelImage');
        switch (model) {
            case 'lstm_1layer_econ':
                imageElement.src = 'static/1-layer LSTM_Economic_Index.png';  
                break;
            case 'lstm_1layer_main':
                imageElement.src = 'static/1-layer LSTM_Main_Index.png';  
                break;
            case 'lstm_1layer_tech':
                imageElement.src = 'static/1-layer LSTM_Technical_Index.png';  
                break;
            case 'lstm_2layer_econ':
                imageElement.src = 'static/2-layer LSTM_Economic_Index.png';  
                break;
            case 'lstm_2layer_main':
                imageElement.src = 'static/2-layer LSTM_Main_Index.png';  
                break;
            case 'lstm_2layer_tech':
                imageElement.src = 'static/2-layer LSTM_Technical_Index.png';  
                break;
            case 'lstm_bi_econ':
                imageElement.src = 'static/BiLSTM_Economic_Index.png';  
                break;
            case 'lstm_bi_main':
                imageElement.src = 'static/BiLSTM_Main_Index.png';  
                break;
            case 'lstm_bi_tech':
                imageElement.src = 'static/BiLSTM_Technical_Index.png';  
                break;  
            case 'trans_main':
                imageElement.src = 'static/Transformer_Main_Index.png';  
                break;    
            case 'trans_tech':
                imageElement.src = 'static/Transformer_Technical_Index.png';  
                break; 
            case 'trans_econ':
                imageElement.src = 'static/Transformer_Economic_Index.png';  
                break; 
            default:
                imageElement.src = 'static/default_img .png';  // Ảnh mặc định
                break;
        }
    })
    .catch(error => console.error('Error:', error));
});
