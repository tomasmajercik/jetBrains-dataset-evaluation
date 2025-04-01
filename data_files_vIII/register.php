<?php
    header("Access-Control-Allow-Origin: http://localhost:3000");
    header("Access-Control-Allow-Headers: Content-Type");
    header("Content-Type: application/json; charset=UTF-8");
    session_start();

    $connection = mysqli_connect("localhost", "root", "", "webologyTask");

    if(mysqli_connect_error()) {
        echo json_encode(["success" => false, "message" => "Failed to connect to MySQL: " . mysqli_connect_error()]);
        die();
    }

    if ($_SERVER["REQUEST_METHOD"] == "POST") 
    {
        $name = mysqli_real_escape_string($connection, $_POST["name"]);
        $password = mysqli_real_escape_string($connection, $_POST["password"]);
        $email = mysqli_real_escape_string($connection, $_POST["email"]);

        // Check if the username or email already exists
        $checkQuery = "SELECT * FROM `users` WHERE `username` = '$name' OR `email` = '$email'";
        $checkResult = mysqli_query($connection, $checkQuery);

        if (mysqli_num_rows($checkResult) > 0) 
        {
            echo json_encode(["success" => false, "message" => "Username or email already exists"]);
        } 
        else 
        {
            $query = "INSERT INTO `users` (username, email, password, token) VALUES ('$name', '$email', '$password', '')";

            $result = mysqli_query($connection, $query);

            if ($result) 
            {
                echo json_encode(["success" => true]);
            } 
            else 
            {
                echo json_encode(["success" => false, "message" => "Failed to register user: " . mysqli_error($connection)]);
            }
        }
    } else {
        echo json_encode(["success" => false, "message" => "Invalid request method"]);
    }

    mysqli_close($connection);
?>
