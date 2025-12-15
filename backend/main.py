# main.py - SkillLens API with Job Creation and Resume Parsing
from fastapi import FastAPI, HTTPException, Depends, status, UploadFile, File
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr
import bcrypt
from jose import JWTError, jwt
from datetime import datetime, timedelta
from typing import Optional, Dict, List
from contextlib import asynccontextmanager
import asyncpg
import os
import re
import tempfile

# ==========================================
# CONFIGURATION
# ==========================================
DB_DSN = os.getenv("DB_DSN", "postgresql://admin:password123@localhost:5432/skilllens")
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-secret-key-change-in-production")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "1440"))

security = HTTPBearer()

# ==========================================
# PASSWORD HASHING
# ==========================================
def verify_password(plain_password: str, hashed_password: str) -> bool:
    return bcrypt.checkpw(plain_password.encode('utf-8'), hashed_password.encode('utf-8'))

def get_password_hash(password: str) -> str:
    password_bytes = password.encode('utf-8')[:72]
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(password_bytes, salt).decode('utf-8')

# ==========================================
# RESUME PARSING UTILITIES
# ==========================================
def extract_text_from_pdf(file_path: str) -> str:
    """Extract text from PDF file using pdfplumber"""
    import pdfplumber
    text = ""
    try:
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        print(f"PDF extraction error: {e}")
    return text

def extract_text_from_docx(file_path: str) -> str:
    """Extract text from DOCX file using python-docx"""
    from docx import Document
    text = ""
    try:
        doc = Document(file_path)
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
    except Exception as e:
        print(f"DOCX extraction error: {e}")
    return text

def parse_resume_text(text: str) -> Dict:
    """Extract structured information from resume text"""
    result = {
        "skills": [],
        "experience_years": 0,
        "education": [],
        "email": None,
        "phone": None,
        "name": None,
        "summary": None
    }
    
    # Extract email
    email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    emails = re.findall(email_pattern, text)
    if emails:
        result["email"] = emails[0]
    
    # Extract phone
    phone_pattern = r'[\+]?[(]?[0-9]{1,3}[)]?[-\s\.]?[(]?[0-9]{1,3}[)]?[-\s\.]?[0-9]{3,6}[-\s\.]?[0-9]{3,6}'
    phones = re.findall(phone_pattern, text)
    if phones:
        result["phone"] = phones[0].strip()
    
    # Common tech skills to look for
    tech_skills = [
        "Python", "JavaScript", "TypeScript", "Java", "C++", "C#", "Go", "Rust", "Ruby",
        "React", "Angular", "Vue", "Flutter", "Swift", "Kotlin", "Node.js", "Django",
        "FastAPI", "Spring", "AWS", "Azure", "GCP", "Docker", "Kubernetes", "Terraform",
        "PostgreSQL", "MySQL", "MongoDB", "Redis", "GraphQL", "REST", "SQL",
        "Machine Learning", "Deep Learning", "TensorFlow", "PyTorch", "AI", "ML",
        "Git", "Linux", "Agile", "Scrum", "CI/CD", "DevOps", "Microservices",
        "HTML", "CSS", "SASS", "Tailwind", "Bootstrap", "Next.js", "Express"
    ]
    
    text_lower = text.lower()
    found_skills = []
    for skill in tech_skills:
        if skill.lower() in text_lower:
            found_skills.append(skill)
    result["skills"] = found_skills[:15]  # Limit to 15 skills
    
    # Extract years of experience
    exp_patterns = [
        r'(\d+)\+?\s*years?\s*(?:of\s*)?(?:experience|exp)',
        r'experience[:\s]*(\d+)\+?\s*years?',
        r'(\d+)\+?\s*years?\s*(?:in|of)\s*(?:software|development|engineering)',
    ]
    for pattern in exp_patterns:
        matches = re.findall(pattern, text_lower)
        if matches:
            try:
                result["experience_years"] = int(matches[0])
                break
            except:
                pass
    
    # Extract education
    education_keywords = ["Bachelor", "Master", "PhD", "B.S.", "M.S.", "B.A.", "M.A.", "MBA", "BSc", "MSc"]
    for keyword in education_keywords:
        if keyword.lower() in text_lower:
            result["education"].append(keyword)
    
    # Try to extract name (first line often contains name)
    lines = text.strip().split('\n')
    if lines:
        first_line = lines[0].strip()
        if len(first_line) < 50 and not '@' in first_line and not any(c.isdigit() for c in first_line):
            result["name"] = first_line
    
    # Extract summary (first paragraph that's longer than 100 chars)
    paragraphs = text.split('\n\n')
    for para in paragraphs:
        clean_para = para.strip()
        if len(clean_para) > 100 and len(clean_para) < 1000:
            result["summary"] = clean_para[:500]
            break
    
    return result

# ==========================================
# LIFESPAN
# ==========================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("\n🚀 BACKEND: Starting Database Connection Pool...")
    try:
        app.state.pool = await asyncpg.create_pool(DB_DSN)
        print("✅ BACKEND: Database Connected Successfully\n")
    except Exception as e:
        print(f"❌ BACKEND: Database Connection Failed: {e}\n")
        raise
    yield
    if hasattr(app.state, 'pool') and app.state.pool:
        await app.state.pool.close()
        print("👋 BACKEND: Database Connection Pool Closed")

app = FastAPI(title="SkillLens API", lifespan=lifespan)

# ==========================================
# MODELS
# ==========================================
class UserRegister(BaseModel):
    email: EmailStr
    password: str
    role: str

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class UserProfileUpdate(BaseModel):
    answers: Dict[str, str]

class TokenResponse(BaseModel):
    access_token: str
    token_type: str
    role: str
    email: str

class JobCreate(BaseModel):
    title: str
    description: Optional[str] = None
    salary_min: Optional[int] = None
    salary_max: Optional[int] = None
    required_skills: Optional[List[str]] = []
    work_mode: Optional[str] = "hybrid"

class JobResponse(BaseModel):
    id: int
    title: str
    description: Optional[str]
    salary_min: Optional[int]
    salary_max: Optional[int]
    required_skills: Optional[List[str]]
    work_mode: Optional[str]
    is_active: bool

class CandidateProfileUpdate(BaseModel):
    skills: Optional[List[str]] = None
    experience_years: Optional[int] = None
    salary_min: Optional[int] = None
    salary_max: Optional[int] = None
    work_mode: Optional[str] = None
    bio: Optional[str] = None
    name: Optional[str] = None

# ==========================================
# AUTH & HELPERS
# ==========================================
def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        user_id: int = payload.get("sub")
        email: str = payload.get("email")
        role: str = payload.get("role")
        
        if user_id is None:
            print("❌ AUTH ERROR: Token missing 'sub' claim")
            raise credentials_exception
            
        return {"id": user_id, "email": email, "role": role}
        
    except JWTError as e:
        print(f"❌ AUTH ERROR: JWT Verification Failed - {str(e)}")
        raise credentials_exception
    except Exception as e:
        print(f"❌ AUTH ERROR: Unexpected error - {str(e)}")
        raise credentials_exception

# ==========================================
# ENDPOINTS
# ==========================================
@app.post("/register", response_model=TokenResponse)
async def register(user: UserRegister):
    pool = app.state.pool
    
    if user.role not in ['candidate', 'recruiter']:
        raise HTTPException(status_code=400, detail="Role must be 'candidate' or 'recruiter'")
    if len(user.password) < 6:
        raise HTTPException(status_code=400, detail="Password too short")
    
    try:
        hashed_password = get_password_hash(user.password)
        
        row = await pool.fetchrow(
            "INSERT INTO users (email, password_hash, role) VALUES ($1, $2, $3) RETURNING id",
            user.email.lower(), hashed_password, user.role
        )
        user_id = row['id']
        
        if user.role == 'candidate':
            await pool.execute("INSERT INTO candidate_profiles (user_id) VALUES ($1)", user_id)
        else:
            await pool.execute("INSERT INTO recruiter_profiles (user_id) VALUES ($1)", user_id)
        
        access_token = create_access_token(
            data={"sub": str(user_id), "email": user.email.lower(), "role": user.role}
        )
        
        return TokenResponse(
            access_token=access_token,
            token_type="bearer",
            role=user.role,
            email=user.email.lower()
        )
        
    except asyncpg.UniqueViolationError:
        raise HTTPException(status_code=400, detail="Email already registered")
    except Exception as e:
        print(f"Register Error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/login", response_model=TokenResponse)
async def login(credentials: UserLogin):
    pool = app.state.pool
    row = await pool.fetchrow(
        "SELECT id, email, password_hash, role, is_active FROM users WHERE email = $1",
        credentials.email.lower()
    )
    
    if not row or not row['is_active'] or not verify_password(credentials.password, row['password_hash']):
        raise HTTPException(status_code=401, detail="Invalid email or password")
    
    access_token = create_access_token(
        data={"sub": str(row['id']), "email": row['email'], "role": row['role']}
    )
    
    return TokenResponse(
        access_token=access_token,
        token_type="bearer",
        role=row['role'],
        email=row['email']
    )

@app.post("/update_profile")
async def update_profile(data: UserProfileUpdate, current_user: dict = Depends(get_current_user)):
    pool = app.state.pool
    user_id = int(current_user['id'])
    role = current_user['role']
    
    print(f"📝 UPDATING PROFILE: User {user_id} ({role})")
    
    try:
        if role == 'candidate':
            tech_skill = data.answers.get("0", "General")
            experience_str = data.answers.get("1", "0")
            salary_str = data.answers.get("2", "0")
            work_mode_str = data.answers.get("3", "hybrid")
            
            try:
                exp_years = int(experience_str.split('-')[0].split('+')[0].strip())
            except:
                exp_years = 0
            
            try:
                salary_min = int(salary_str.replace('$', '').replace('k', '000').split('-')[0].strip())
            except:
                salary_min = 50000
                
            work_mode_map = {"Remote Only": "remote", "Hybrid": "hybrid", "On-site": "onsite"}
            work_mode = work_mode_map.get(work_mode_str, "hybrid")
            
            await pool.execute("""
                UPDATE candidate_profiles 
                SET skills = ARRAY[$1], experience_years = $2, salary_min = $3, work_mode = $4, updated_at = NOW()
                WHERE user_id = $5
            """, tech_skill, exp_years, salary_min, work_mode, user_id)
            
            await pool.execute(
                "INSERT INTO skill_trends (time, skill_name, source, weight) VALUES (NOW(), $1, 'profile_update', 1)",
                tech_skill
            )
            
        else:
            industry = data.answers.get("0", "Technology")
            priority = data.answers.get("1", "Quality")
            team_size = data.answers.get("2", "11-50")
            
            await pool.execute("""
                UPDATE recruiter_profiles 
                SET industry = $1, hiring_priority = $2, team_size = $3, updated_at = NOW()
                WHERE user_id = $4
            """, industry, priority, team_size, user_id)
        
        return {"status": "updated", "role": role}
    except Exception as e:
        print(f"❌ UPDATE FAILED: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/update_candidate_profile")
async def update_candidate_profile(data: CandidateProfileUpdate, current_user: dict = Depends(get_current_user)):
    """Update candidate profile with structured data (from resume parsing)"""
    pool = app.state.pool
    user_id = int(current_user['id'])
    
    if current_user['role'] != 'candidate':
        raise HTTPException(status_code=403, detail="Only candidates can update candidate profiles")
    
    try:
        update_fields = []
        values = []
        param_count = 1
        
        if data.skills is not None:
            update_fields.append(f"skills = ${param_count}")
            values.append(data.skills)
            param_count += 1
            
            # Add skills to trends
            for skill in data.skills[:5]:
                await pool.execute(
                    "INSERT INTO skill_trends (time, skill_name, source, weight) VALUES (NOW(), $1, 'resume_upload', 1)",
                    skill
                )
        
        if data.experience_years is not None:
            update_fields.append(f"experience_years = ${param_count}")
            values.append(data.experience_years)
            param_count += 1
        
        if data.salary_min is not None:
            update_fields.append(f"salary_min = ${param_count}")
            values.append(data.salary_min)
            param_count += 1
        
        if data.salary_max is not None:
            update_fields.append(f"salary_max = ${param_count}")
            values.append(data.salary_max)
            param_count += 1
        
        if data.work_mode is not None:
            update_fields.append(f"work_mode = ${param_count}")
            values.append(data.work_mode)
            param_count += 1
        
        if data.bio is not None:
            update_fields.append(f"bio = ${param_count}")
            values.append(data.bio)
            param_count += 1
        
        if update_fields:
            update_fields.append("updated_at = NOW()")
            values.append(user_id)
            
            query = f"""
                UPDATE candidate_profiles 
                SET {', '.join(update_fields)}
                WHERE user_id = ${param_count}
            """
            await pool.execute(query, *values)
        
        return {"status": "updated", "fields_updated": len(update_fields) - 1}
    except Exception as e:
        print(f"❌ CANDIDATE PROFILE UPDATE FAILED: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload_resume")
async def upload_resume(file: UploadFile = File(...), current_user: dict = Depends(get_current_user)):
    """Upload and parse resume (PDF or DOCX)"""
    if current_user['role'] != 'candidate':
        raise HTTPException(status_code=403, detail="Only candidates can upload resumes")
    
    filename = file.filename.lower()
    if not (filename.endswith('.pdf') or filename.endswith('.docx') or filename.endswith('.doc')):
        raise HTTPException(status_code=400, detail="Only PDF and DOCX files are supported")
    
    try:
        # Save uploaded file temporarily
        suffix = '.pdf' if filename.endswith('.pdf') else '.docx'
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        # Extract text based on file type
        if filename.endswith('.pdf'):
            text = extract_text_from_pdf(tmp_path)
        else:
            text = extract_text_from_docx(tmp_path)
        
        # Clean up temp file
        os.unlink(tmp_path)
        
        if not text.strip():
            raise HTTPException(status_code=400, detail="Could not extract text from file")
        
        # Parse the extracted text
        parsed_data = parse_resume_text(text)
        
        return {
            "status": "success",
            "filename": file.filename,
            "extracted_data": parsed_data,
            "raw_text_preview": text[:500] if len(text) > 500 else text
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ RESUME UPLOAD ERROR: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

@app.get("/profile")
async def get_profile(current_user: dict = Depends(get_current_user)):
    pool = app.state.pool
    user_id = int(current_user['id'])
    table = "candidate_profiles" if current_user['role'] == 'candidate' else "recruiter_profiles"
    
    row = await pool.fetchrow(f"SELECT * FROM {table} WHERE user_id = $1", user_id)
    return dict(row) if row else {}

# ==========================================
# JOB ENDPOINTS
# ==========================================
@app.post("/jobs", response_model=JobResponse)
async def create_job(job: JobCreate, current_user: dict = Depends(get_current_user)):
    """Create a new job posting (recruiters only)"""
    if current_user['role'] != 'recruiter':
        raise HTTPException(status_code=403, detail="Only recruiters can create jobs")
    
    pool = app.state.pool
    user_id = int(current_user['id'])
    
    try:
        row = await pool.fetchrow("""
            INSERT INTO jobs (recruiter_id, title, description, salary_min, salary_max, required_skills, work_mode)
            VALUES ($1, $2, $3, $4, $5, $6, $7)
            RETURNING id, title, description, salary_min, salary_max, required_skills, work_mode, is_active
        """, user_id, job.title, job.description, job.salary_min, job.salary_max, 
             job.required_skills or [], job.work_mode or 'hybrid')
        
        # Add required skills to trends
        if job.required_skills:
            for skill in job.required_skills[:5]:
                await pool.execute(
                    "INSERT INTO skill_trends (time, skill_name, source, weight) VALUES (NOW(), $1, 'job_posting', 2)",
                    skill
                )
        
        return JobResponse(**dict(row))
    except Exception as e:
        print(f"❌ JOB CREATE ERROR: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/jobs")
async def get_jobs(current_user: dict = Depends(get_current_user)):
    """Get jobs - for candidates: all active jobs; for recruiters: their own jobs"""
    pool = app.state.pool
    user_id = int(current_user['id'])
    
    if current_user['role'] == 'recruiter':
        rows = await pool.fetch("""
            SELECT j.*, 
                   (SELECT COUNT(*) FROM applications a WHERE a.job_id = j.id) as applicant_count
            FROM jobs j 
            WHERE j.recruiter_id = $1 
            ORDER BY j.created_at DESC
        """, user_id)
    else:
        rows = await pool.fetch("""
            SELECT j.*, 
                   u.email as recruiter_email,
                   rp.company_name,
                   (SELECT COUNT(*) FROM applications a WHERE a.job_id = j.id) as applicant_count,
                   EXISTS(SELECT 1 FROM applications a WHERE a.job_id = j.id AND a.candidate_id = $1) as has_applied,
                   EXISTS(SELECT 1 FROM recruiter_interests ri WHERE ri.job_id = j.id AND ri.candidate_id = $1) as recruiter_interested
            FROM jobs j
            JOIN users u ON j.recruiter_id = u.id
            LEFT JOIN recruiter_profiles rp ON rp.user_id = j.recruiter_id
            WHERE j.is_active = TRUE
            ORDER BY j.created_at DESC
            LIMIT 50
        """, user_id)
    
    return [dict(r) for r in rows]

@app.put("/jobs/{job_id}")
async def update_job(job_id: int, job: JobCreate, current_user: dict = Depends(get_current_user)):
    """Update a job posting"""
    if current_user['role'] != 'recruiter':
        raise HTTPException(status_code=403, detail="Only recruiters can update jobs")
    
    pool = app.state.pool
    user_id = int(current_user['id'])
    
    # Verify ownership
    existing = await pool.fetchrow("SELECT id FROM jobs WHERE id = $1 AND recruiter_id = $2", job_id, user_id)
    if not existing:
        raise HTTPException(status_code=404, detail="Job not found or you don't have permission")
    
    await pool.execute("""
        UPDATE jobs 
        SET title = $1, description = $2, salary_min = $3, salary_max = $4, 
            required_skills = $5, work_mode = $6, updated_at = NOW()
        WHERE id = $7
    """, job.title, job.description, job.salary_min, job.salary_max, 
         job.required_skills or [], job.work_mode, job_id)
    
    return {"status": "updated", "job_id": job_id}

@app.delete("/jobs/{job_id}")
async def delete_job(job_id: int, current_user: dict = Depends(get_current_user)):
    """Delete (deactivate) a job posting"""
    if current_user['role'] != 'recruiter':
        raise HTTPException(status_code=403, detail="Only recruiters can delete jobs")
    
    pool = app.state.pool
    user_id = int(current_user['id'])
    
    result = await pool.execute(
        "UPDATE jobs SET is_active = FALSE WHERE id = $1 AND recruiter_id = $2",
        job_id, user_id
    )
    
    return {"status": "deleted", "job_id": job_id}

# ==========================================
# APPLICATION ENDPOINTS
# ==========================================
@app.post("/apply/{job_id}")
async def apply_to_job(job_id: int, current_user: dict = Depends(get_current_user)):
    """Apply to a job (candidates only)"""
    if current_user['role'] != 'candidate':
        raise HTTPException(status_code=403, detail="Only candidates can apply to jobs")
    
    pool = app.state.pool
    user_id = int(current_user['id'])
    
    # Check if job exists and is active
    job = await pool.fetchrow("SELECT id FROM jobs WHERE id = $1 AND is_active = TRUE", job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found or inactive")
    
    try:
        await pool.execute("""
            INSERT INTO applications (candidate_id, job_id, status)
            VALUES ($1, $2, 'pending')
            ON CONFLICT (candidate_id, job_id) DO NOTHING
        """, user_id, job_id)
        return {"status": "applied", "job_id": job_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/interest/{candidate_id}/{job_id}")
async def express_interest(candidate_id: int, job_id: int, current_user: dict = Depends(get_current_user)):
    """Express interest in a candidate for a job (recruiters only)"""
    if current_user['role'] != 'recruiter':
        raise HTTPException(status_code=403, detail="Only recruiters can express interest")
    
    pool = app.state.pool
    user_id = int(current_user['id'])
    
    # Verify the recruiter owns this job
    job = await pool.fetchrow("SELECT id FROM jobs WHERE id = $1 AND recruiter_id = $2", job_id, user_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found or you don't have permission")
    
    try:
        await pool.execute("""
            INSERT INTO recruiter_interests (recruiter_id, candidate_id, job_id)
            VALUES ($1, $2, $3)
            ON CONFLICT (recruiter_id, candidate_id, job_id) DO NOTHING
        """, user_id, candidate_id, job_id)
        return {"status": "interest_expressed", "candidate_id": candidate_id, "job_id": job_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ==========================================
# MATCHING & DISCOVERY
# ==========================================
@app.get("/matches")
async def get_matches(current_user: dict = Depends(get_current_user)):
    """Get AI-powered matches"""
    pool = app.state.pool
    user_id = int(current_user['id'])
    
    if current_user['role'] == 'candidate':
        # Get jobs that match candidate's profile
        rows = await pool.fetch("""
            SELECT j.id, j.title as role, 
                   COALESCE(rp.company_name, 'Company') as company,
                   CONCAT('$', j.salary_min/1000, 'k - $', j.salary_max/1000, 'k') as salary,
                   j.work_mode,
                   EXISTS(SELECT 1 FROM recruiter_interests ri WHERE ri.job_id = j.id AND ri.candidate_id = $1) as recruiter_interested,
                   EXISTS(SELECT 1 FROM applications a WHERE a.job_id = j.id AND a.candidate_id = $1) as has_applied
            FROM jobs j
            LEFT JOIN recruiter_profiles rp ON rp.user_id = j.recruiter_id
            WHERE j.is_active = TRUE
            ORDER BY j.created_at DESC
            LIMIT 20
        """, user_id)
        
        return [dict(r) for r in rows]
    else:
        # Get candidates that have applied to recruiter's jobs
        rows = await pool.fetch("""
            SELECT DISTINCT ON (u.id)
                   u.id as candidate_id,
                   u.email,
                   cp.skills,
                   cp.experience_years,
                   cp.work_mode,
                   cp.bio,
                   a.job_id,
                   j.title as job_title,
                   a.status as application_status
            FROM applications a
            JOIN users u ON a.candidate_id = u.id
            JOIN candidate_profiles cp ON cp.user_id = u.id
            JOIN jobs j ON a.job_id = j.id
            WHERE j.recruiter_id = $1
            ORDER BY u.id, a.created_at DESC
            LIMIT 20
        """, user_id)
        
        return [dict(r) for r in rows]

@app.get("/candidates")
async def get_candidates(current_user: dict = Depends(get_current_user)):
    """Get all candidates (recruiters only)"""
    if current_user['role'] != 'recruiter':
        raise HTTPException(status_code=403, detail="Only recruiters can view candidates")
    
    pool = app.state.pool
    
    rows = await pool.fetch("""
        SELECT u.id, u.email, cp.skills, cp.experience_years, cp.work_mode, cp.bio
        FROM users u
        JOIN candidate_profiles cp ON cp.user_id = u.id
        WHERE u.role = 'candidate' AND u.is_active = TRUE
        ORDER BY cp.updated_at DESC NULLS LAST
        LIMIT 50
    """)
    
    return [dict(r) for r in rows]

@app.get("/trends")
async def get_trends(current_user: dict = Depends(get_current_user)):
    """Get skill trends from actual data"""
    pool = app.state.pool
    rows = await pool.fetch("""
        SELECT skill_name as text, SUM(weight) as size 
        FROM skill_trends 
        WHERE time > NOW() - INTERVAL '30 days'
        GROUP BY skill_name 
        ORDER BY size DESC
        LIMIT 15
    """)
    
    if not rows:
        return []
    
    return [{"text": r['text'], "size": int(r['size']) * 10} for r in rows]

@app.get("/health")
async def health_check():
    try:
        await app.state.pool.fetchval("SELECT 1")
        return {"status": "healthy", "database": "connected"}
    except:
        return {"status": "unhealthy", "database": "disconnected"}