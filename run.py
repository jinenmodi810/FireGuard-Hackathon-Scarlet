from src import create_app

# Create the Flask app from your factory
app = create_app()

if __name__ == "__main__":
    app.run(debug=True)
