<?php
    header("Access-Control-Allow-Origin: http://localhost:3000");
    header("Access-Control-Allow-Headers: Content-Type");
    header("Content-Type: application/json; charset=UTF-8");
    session_start();

    $connection = mysqli_connect("localhost", "root", "", "webologyTask");

    if (mysqli_connect_error()) 
    {
        echo json_encode(["success" => false, "message" => mysqli_connect_error()]);
        die();
    }

    if ($_SERVER["REQUEST_METHOD"] == "POST" && isset($_POST["token"])) 
    {
        $token = mysqli_real_escape_string($connection, $_POST["token"]);

        $query = "SELECT * FROM `users` WHERE `token` = '$token'";
        $result = mysqli_query($connection, $query);

        if (mysqli_num_rows($result) > 0) 
        {
            $row = mysqli_fetch_assoc($result);
            $username = $row["username"];

            $fileQuery = "SELECT * FROM `user_files` WHERE `username` = '$username'";
            $filesResult = mysqli_query($connection, $fileQuery);

            $files = [];
            while ($file = mysqli_fetch_assoc($filesResult))
            {
                $files[] = $file;
            }

            echo json_encode(["success" => true, "username" => $username, "files" => $files]);
        } 
        else 
        {
            echo json_encode(["success" => false, "message" => "Invalid token"]);
        }
    } else {
        echo json_encode(["success" => false, "message" => "Invalid request"]);
    }
?>
