var train_graph;
var interval_id;

window.addEventListener('DOMContentLoaded', (event)=>{
   console.log("Event: ", event);

   // Add Button Event Listener
    train_button = document.getElementById('train_go');
    train_button.addEventListener('click', (event)=>{
        console.log("Train!");
        // Do ajax call
        $.ajax({
            url: '/train',
            type: 'GET',
            dataType: 'json',
            success: (response)=>{
                console.log("Response: ", response);
            },
            error: (err)=>{
                console.log("Err: ", err);
            }
        });
        interval_id = setInterval(check_status, 5000);
    });

    // Setup basic graph
    training_graph = document.getElementById('training_graph');
    chart2d = training_graph.getContext('2d');
    train_graph = new Chart(chart2d, {
        type: 'line',
        data: {
            labels: [2, 3, 4, 5],
            datasets: [{
                label: 'Train Loss',
                yAxisID: 'Loss',
                fill: false,
                backgroundColor: "rgba(20, 120, 255, 1)",
                borderColor: "rgba(20, 120, 255, 1)",
            },
            {
                label: 'Test Loss',
                yAxisID: 'Test Loss',
                fill: false,
                backgroundColor: "rgba(255, 50, 100, 1)",
                borderColor: "rgba(255, 50, 100, 1)",
            }]
        },
        options:{
            scales:{
                yAxes: [{
                    name: 'Loss',
                    id: 'Loss',
                    position: 'left'
                },
                {
                    name: 'Test Loss',
                    id: 'Test Loss',
                    position: 'right'
                }]
            }
        }
    });

});

function check_status(){
    // Runs every 5 seconds
    // Get Latest Data
    $.ajax({
        url:'/train/data/',
        type:'GET',
        dataType: 'json',
        success: (response)=>{
            epoch_losses = response.epoch_loss;
            epoch_test_losses = response.epoch_test_losses;
            test_accs = response.test_accs;
            curr_active = response.curr_active;
            if(curr_active==false){
                clearInterval(interval_id);
            }
            training_graph = document.getElementById('training_graph');
            chart2d = training_graph.getContext('2d');
            x_axis = [];
            for(i=0;i<epoch_losses.length;i++){
                x_axis.push(i);
            }
            train_graph.data.labels = x_axis;
            train_graph.data.datasets[0].data = epoch_losses;
            train_graph.data.datasets[1].data = epoch_test_losses;
            train_graph.update();
        }
    })
}