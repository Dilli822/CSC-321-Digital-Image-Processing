def generate_html_from_txt(txt_path, html_path):
    try:
        # Read the contents of the .txt file
        with open(txt_path, 'r', encoding='utf-8') as file:
            report_content = file.read()
        
        # Bootstrap styled HTML template
        html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta http-equiv="X-UA-Compatible" content="IE=edge">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Medical Report</title>
            <!-- Bootstrap CSS -->
            <link href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css" rel="stylesheet">
        </head>
        <body>
            <div class="container mt-5">
                <h1 class="text-center mb-4">Medical Report</h1>
                <div class="card">
                    <div class="card-body">
                        <pre class="text-left" style="white-space: pre-wrap; font-size: 1rem; line-height: 1.6;">
{report_content}
                        </pre>
                    </div>
                </div>
            </div>
            <!-- Bootstrap JS and dependencies -->
            <script src="https://code.jquery.com/jquery-3.5.1.slim.min.js"></script>
            <script src="https://cdn.jsdelivr.net/npm/@popperjs/core@2.9.2/dist/umd/popper.min.js"></script>
            <script src="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/js/bootstrap.min.js"></script>
        </body>
        </html>
        """
        
        # Write the HTML content to the output HTML file
        with open(html_path, 'w', encoding='utf-8') as html_file:
            html_file.write(html_content)
        
        print(f"HTML file generated: {html_path}")

    except FileNotFoundError:
        print(f"Error: The file {txt_path} does not exist.")
    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage
generate_html_from_txt('medical_report.txt', 'medical_report.html')
