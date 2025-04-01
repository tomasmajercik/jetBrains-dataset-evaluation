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

        if ($_SERVER["REQUEST_METHOD"] == "POST" && !empty($_FILES["file"])) {
            $username = mysqli_real_escape_string($connection, $_POST["username"]);
            $files = $_FILES["file"];
            $fileCount = count($files["name"]);
        
            $responses = [];
        
            if (!is_dir("storage")) {
                mkdir("storage");
            }
        
            for ($i = 0; $i < $fileCount; $i++) {
                $fileName = basename($files["name"][$i]);
                $filePath = "storage/" . $fileName;
        
                // Check if the file has been uploaded correctly
                if ($files['error'][$i] !== UPLOAD_ERR_OK) {
                    $responses[] = ["file" => $fileName, "success" => false, "message" => "File upload error: " . $files['error'][$i]];
                    continue;
                }
        
                if (!is_writable("storage")) {
                    echo json_encode(["success" => false, "message" => "Storage directory is not writable"]);
                    exit;
                }
        
                if (move_uploaded_file($files["tmp_name"][$i], $filePath)) {
                    $query = "INSERT INTO `user_files` (username, file_name, file_path) VALUES ('$username', '$fileName', '$filePath')";
                    if (mysqli_query($connection, $query)) {
                        $responses[] = ["file" => $fileName, "success" => true, "message" => "File uploaded successfully!"];
                    } else {
                        $responses[] = ["file" => $fileName, "success" => false, "message" => "Failed to save file information to the database"];
                    }
                } else {
                    $responses[] = ["file" => $fileName, "success" => false, "message" => "Failed to move uploaded file"];
                }
            }
        
            echo json_encode(["success" => true, "responses" => $responses]);
        } else {
            echo json_encode(["success" => false, "message" => "Invalid request"]);
        }
?>
