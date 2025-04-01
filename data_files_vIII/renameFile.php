<?php
    header("Access-Control-Allow-Origin: http://localhost:3000");
    header("Access-Control-Allow-Headers: Content-Type");
    header("Content-Type: application/json; charset=UTF-8");
    session_start();

    $connection = mysqli_connect("localhost", "root", "", "webologyTask");

    
        if (mysqli_connect_error()) 
        {
            error_log("Connection failed: " . mysqli_connect_error());
            echo json_encode(["success" => false, "message" => "Database connection failed"]);
            die();
        }

        if ($_SERVER["REQUEST_METHOD"] == "POST")
        {
            $username = mysqli_real_escape_string($connection, $_POST["username"]);
            $oldName = $_POST["oldName"];
            $newName = $_POST["newName"];
        
            $query = "UPDATE user_files SET file_name='$newName' WHERE file_name='$oldName' AND username='$username'";
            if(mysqli_query($connection, $query))
            {
                echo json_encode(["success" => true]);
            }
            else{
                echo json_encode(["success" => false]);
            }
        
        } else {
            echo json_encode(["success" => false, "message" => "Invalid request"]);
        }
?>
