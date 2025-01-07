pipeline {
    agent any
 
    environment {
        PYTHON_PATH = 'C:/Program Files/Python313/;C:/Program Files/Python313/Scripts/'
        SONAR_SCANNER_PATH = 'C:/Users/Sahil Makhija/Downloads/sonar-scanner-cli-6.2.1.4610-windows-x64/sonar-scanner-6.2.1.4610-windows-x64/bin'
    }
 
    stages {
        stage('Checkout') {
            steps {
                checkout scm
            }
        }
 
        stage('Build') {
            steps {
                // Verify Python and pip paths and install dependencies
                bat '''
                echo Validating Python and pip installations...
                set PATH=%PYTHON_PATH%;%PATH%
                python --version || echo "Python not found. Check PYTHON_PATH."
                pip --version || echo "Pip not found. Check PYTHON_PATH."
                
                echo Installing dependencies from requirements.txt...
                if exist requirements.txt (
                    pip install --upgrade pip
                    pip install -r requirements.txt || exit /b 1
                ) else (
                    echo "requirements.txt not found. Exiting build stage."
                    exit /b 1
                )
                '''
            }
        }
 
        stage('SonarQube Analysis') {
            environment {
                SONAR_TOKEN = credentials('sonar-token') // Accessing the SonarQube token stored in Jenkins credentials
            }
            steps {
                // Ensure SonarQube Scanner is available
                bat '''
                set PATH=%SONAR_SCANNER_PATH%;%PATH%
                where sonar-scanner || echo "SonarQube scanner not found. Please install it."
                set PATH=%PYTHON_PATH%;%PATH%
                sonar-scanner -Dsonar.projectKey=jump-counter ^
                    -Dsonar.sources=. ^
                    -Dsonar.host.url=http://localhost:9000 ^
                    -Dsonar.token=%SONAR_TOKEN%
                '''
            }
        }
    }
 
    post {
        success {
            echo 'Pipeline completed successfully'
        }
        failure {
            echo 'Pipeline failed'
        }
        always {
            echo 'This runs regardless of the result.'
        }
    }
}
