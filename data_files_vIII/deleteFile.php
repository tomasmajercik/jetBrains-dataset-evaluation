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
            $fileName = $_POST["fileName"];

            
            $query = "SELECT `file_path` FROM `user_files` WHERE file_name='$fileName' AND username='$username'";
            $result = mysqli_query($connection, $query);
        
            if ($result) {
                $row = mysqli_fetch_assoc($result);
                $filePath = $row['file_path'];
                
                $deleteQuery = "DELETE FROM `user_files` WHERE file_name='$fileName' AND username='$username'";
                
                if (mysqli_query($connection, $deleteQuery)) 
                {
                    if (file_exists($filePath)) 
                    {
                        if (unlink($filePath)) 
                        {
                            echo json_encode(["success" => true]);
                        }
                        else 
                        {
                            echo json_encode(["success" => false, "message" => "Failed to delete the file from the filesystem"]);
                        }
                    } 
                    else 
                    {
                        echo json_encode(["success" => false, "message" => "File does not exist on the server"]);
                    }
                } 
                else 
                {
                    echo json_encode(["success" => false, "message" => "Failed to delete file from the database"]);
                }
            } 
            else 
            {
                echo json_encode(["success" => false, "message" => "Failed to fetch file path from the database"]);
            }
        } 
        else 
        {
            echo json_encode(["success" => false, "message" => "Invalid request"]);
        }
?>
        