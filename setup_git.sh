!#/bin/bash
# 1. Initialize repository
git init
git add .
git commit -m "Initial commit: Project structure with all three modules"

# 2. Create feature branch
git checkout -b feature/complete-project

# 3. Make first change and commit
echo "# Project Updates" > CHANGELOG.md
git add CHANGELOG.md
git commit -m "Add CHANGELOG documenting project modules"

# 4. Make second change and commit
echo "" >> README.md
echo "## Project Status" >> README.md
echo "All modules complete" >> README.md
git add README.md
git commit -m "Update README with project status"

# 5. Merge back to main
git checkout main
git merge feature/complete-project -m "Merge feature/complete-project"

# 6. Verify
git log --graph --all --oneline