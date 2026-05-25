// ============================================
// TOPOGRAPHIC CANVAS ANIMATION
// Multi-Zone Perlin Noise + Marching Squares
// ============================================

(function() {
    const canvas = document.getElementById('canvas');
    if (!canvas) return; // Exit if canvas not found
    
    const ctx = canvas.getContext('2d');
    let width, height;
    
    // CONFIGURATION
    const CONFIG = {
        gridSize: 18,
        isoLevels: 8,
        baseLineWidth: 2.5,
        zoneCount: 5,
        evolutionSpeed: 0.0005
    };

    // --- FAST NOISE (Permutation Table) ---
    const PERM = new Uint8Array(512);
    const GRAD3 = [[1,1,0],[-1,1,0],[1,-1,0],[-1,-1,0],[1,0,1],[-1,0,1],[1,0,-1],[-1,0,-1],[0,1,1],[0,-1,1],[0,1,-1],[0,-1,-1]];
    
    for(let i=0; i<256; i++) PERM[i] = PERM[i+256] = Math.floor(Math.random() * 256);

    function noise3D(x, y, z) {
        let X = Math.floor(x) & 255, Y = Math.floor(y) & 255, Z = Math.floor(z) & 255;
        x -= Math.floor(x); y -= Math.floor(y); z -= Math.floor(z);
        let u = fade(x), v = fade(y), w = fade(z);
        let A = PERM[X]+Y, AA = PERM[A]+Z, AB = PERM[A+1]+Z, B = PERM[X+1]+Y, BA = PERM[B]+Z, BB = PERM[B+1]+Z;
        
        return lerp(w, lerp(v, lerp(u, grad(PERM[AA], x, y, z), grad(PERM[BA], x-1, y, z)),
                              lerp(u, grad(PERM[AB], x, y-1, z), grad(PERM[BB], x-1, y-1, z))),
                      lerp(v, lerp(u, grad(PERM[AA+1], x, y, z-1), grad(PERM[BA+1], x-1, y, z-1)),
                              lerp(u, grad(PERM[AB+1], x, y-1, z-1), grad(PERM[BB+1], x-1, y-1, z-1))));
    }
    function fade(t) { return t * t * t * (t * (t * 6 - 15) + 10); }
    function lerp(t, a, b) { return a + t * (b - a); }
    function grad(hash, x, y, z) {
        let h = hash & 15;
        let u = h<8 ? x : y, v = h<4 ? y : h==12||h==14 ? x : z;
        return ((h&1) == 0 ? u : -u) + ((h&2) == 0 ? v : -v);
    }

    // --- TERRAIN ZONES ---
    class Zone {
        constructor() {
            this.init();
        }

        init() {
            this.x = Math.random() * width;
            this.y = Math.random() * height;
            this.vx = (Math.random() - 0.5) * 0.05;
            this.vy = (Math.random() - 0.5) * 0.05;
            this.radius = Math.min(width, height) * (0.4 + Math.random() * 0.4);
            this.noiseOffset = Math.random() * 1000;
            this.noiseScale = 0.002 + Math.random() * 0.003;
            this.timeSpeed = CONFIG.evolutionSpeed * (0.8 + Math.random() * 0.4);
            this.t = 0;
        }

        update() {
            this.x += this.vx;
            this.y += this.vy;
            this.t += this.timeSpeed;

            if (this.x < -this.radius) this.vx = Math.abs(this.vx);
            if (this.x > width + this.radius) this.vx = -Math.abs(this.vx);
            if (this.y < -this.radius) this.vy = Math.abs(this.vy);
            if (this.y > height + this.radius) this.vy = -Math.abs(this.vy);
        }

        evaluate(px, py) {
            const dx = px - this.x;
            const dy = py - this.y;
            const distSq = dx*dx + dy*dy;
            const rSq = this.radius * this.radius;

            if (distSq > rSq) return 0;

            const falloff = Math.pow(1 - distSq / rSq, 2);
            const n = noise3D(px * this.noiseScale, py * this.noiseScale, this.t + this.noiseOffset);
            
            return (0.5 + 0.5 * n) * falloff;
        }
    }

    // --- MARCHING SQUARES STATE ---
    let zones = [];
    let cols, rows;
    let grid = [];

    function resize() {
        width = canvas.width = window.innerWidth;
        height = canvas.height = window.innerHeight;
        
        cols = Math.ceil(width / CONFIG.gridSize) + 1;
        rows = Math.ceil(height / CONFIG.gridSize) + 1;
        grid = new Float32Array(cols * rows);

        if (zones.length === 0) {
            for (let i = 0; i < CONFIG.zoneCount; i++) zones.push(new Zone());
        }
    }

    function updateGrid() {
        zones.forEach(z => z.update());

        let idx = 0;
        for (let y = 0; y < rows; y++) {
            for (let x = 0; x < cols; x++) {
                const px = x * CONFIG.gridSize;
                const py = y * CONFIG.gridSize;
                
                let sum = 0;
                for (let z = 0; z < zones.length; z++) {
                    sum += zones[z].evaluate(px, py);
                }
                
                grid[idx++] = sum;
            }
        }
    }

    function lerpPos(val1, val2, target) {
        return (target - val1) / (val2 - val1);
    }

    function drawContours() {
        ctx.clearRect(0, 0, width, height);

        for (let k = 1; k <= CONFIG.isoLevels; k++) {
            const threshold = (k / (CONFIG.isoLevels + 1)) * 1.5;
            
            ctx.beginPath();
            
            const opacity = 0.15 + (k / CONFIG.isoLevels) * 0.35;
            ctx.strokeStyle = `rgba(212, 165, 116, ${opacity})`;
            ctx.lineWidth = CONFIG.baseLineWidth;

            for (let y = 0; y < rows - 1; y++) {
                for (let x = 0; x < cols - 1; x++) {
                    const iTL = y * cols + x;
                    const iTR = iTL + 1;
                    const iBR = iTR + cols;
                    const iBL = iTL + cols;

                    const vTL = grid[iTL];
                    const vTR = grid[iTR];
                    const vBR = grid[iBR];
                    const vBL = grid[iBL];

                    let state = 0;
                    if (vTL > threshold) state |= 8;
                    if (vTR > threshold) state |= 4;
                    if (vBR > threshold) state |= 2;
                    if (vBL > threshold) state |= 1;

                    if (state === 0 || state === 15) continue;

                    const cx = x * CONFIG.gridSize;
                    const cy = y * CONFIG.gridSize;
                    const gs = CONFIG.gridSize;

                    const a = { x: cx + gs * lerpPos(vTL, vTR, threshold), y: cy };
                    const b = { x: cx + gs, y: cy + gs * lerpPos(vTR, vBR, threshold) };
                    const c = { x: cx + gs * lerpPos(vBL, vBR, threshold), y: cy + gs };
                    const d = { x: cx, y: cy + gs * lerpPos(vTL, vBL, threshold) };

                    switch (state) {
                        case 1:  drawLine(c, d); break;
                        case 2:  drawLine(b, c); break;
                        case 3:  drawLine(b, d); break;
                        case 4:  drawLine(a, b); break;
                        case 5:  drawLine(a, d); drawLine(b, c); break;
                        case 6:  drawLine(a, c); break;
                        case 7:  drawLine(a, d); break;
                        case 8:  drawLine(d, a); break;
                        case 9:  drawLine(c, a); break;
                        case 10: drawLine(a, b); drawLine(c, d); break;
                        case 11: drawLine(b, a); break;
                        case 12: drawLine(d, b); break;
                        case 13: drawLine(c, b); break;
                        case 14: drawLine(d, c); break;
                    }
                }
            }
            ctx.stroke();
        }
    }

    function drawLine(p1, p2) {
        ctx.moveTo(p1.x, p1.y);
        ctx.lineTo(p2.x, p2.y);
    }

    function animate() {
        updateGrid();
        drawContours();
        requestAnimationFrame(animate);
    }

    window.addEventListener('resize', resize);
    resize();
    animate();
})();

// ============================================
// HEADER SCROLL EFFECT
// ============================================
window.addEventListener('scroll', function() {
    const header = document.getElementById('header');
    if (header) {
        if (window.scrollY > 50) {
            header.classList.add('scrolled');
        } else {
            header.classList.remove('scrolled');
        }
    }
});

// ============================================
// SMOOTH SCROLL FOR ANCHOR LINKS & MOBILE MENU
// ============================================
document.addEventListener('DOMContentLoaded', function() {
    // Smooth scroll
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function(e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth'
                });
            }
        });
    });

    // Mobile Menu Toggle
    const menuToggle = document.getElementById('menuToggle');
    const navLinks = document.querySelector('.nav-links');
    
    if (menuToggle && navLinks) {
        menuToggle.addEventListener('click', function(e) {
            e.stopPropagation();
            menuToggle.classList.toggle('active');
            navLinks.classList.toggle('active');
            document.body.classList.toggle('menu-open');
        });

        // Close menu when a link is clicked
        navLinks.querySelectorAll('a').forEach(link => {
            link.addEventListener('click', function() {
                menuToggle.classList.remove('active');
                navLinks.classList.remove('active');
                document.body.classList.remove('menu-open');
            });
        });

        // Close menu when clicking outside
        document.addEventListener('click', function(e) {
            if (!menuToggle.contains(e.target) && !navLinks.contains(e.target)) {
                menuToggle.classList.remove('active');
                navLinks.classList.remove('active');
                document.body.classList.remove('menu-open');
            }
        });
    }
});
