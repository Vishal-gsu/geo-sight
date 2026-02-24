Write-Host "====== 🚀 Starting GeoSight Project Initialization ======"
Write-Host "[1/4] 📦 Creating/Updating Python Virtual Environment (.venv)..."
python -m venv .venv

Write-Host "[2/4] 🔄 Activating Virtual Environment..."
& .\.venv\Scripts\Activate.ps1

Write-Host "[3/4] 📥 Installing Dependencies (This may take a few minutes)..."
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

Write-Host "[4/4] ⚙️ Setting up .env file..."
Copy-Item -Path ".env.example" -Destination ".env" -ErrorAction SilentlyContinue

Write-Host "====== 🎉 Setup Complete! ======"
Write-Host "⚠️ ACTION REQUIRED: Please open the generated .env file and add your API credentials!"
