%-------SIMULATION PARAMETERS-------
more off;
Nx = 21; %Number of lattices in the x-direction.
Ny = Nx; %Number of lattices in the y-direction.
Re = 10; %Reynolds number.
tau = 0.65; %Relaxation time.
omega = 1/tau; %Relaxation frequency.
vlat = (tau-0.5)/3; %Lattice kinematic viscosity.
Length = Nx-1; %Length of the computational domain.
Ulat = Re*vlat/Length; %Lattice characteristic velocity.
Kx = 2*pi/Length; %Wavenumber in the x- and y-direction.
Ro = 1; %Arbitrary, initial fluid density.
CsSquare = 1/3; %Square of the speed of sound in lattice units.
Ct = 2*Ulat/Length; %Conversion factor for time.
Tsim = 20; %Number of simulation iterations.
%-------LATTICE ARRANGEMENT PARAMETERS (D2Q9)-------
Nl = 9; %Number of lattice linkages.
weight = [4/9,1/9,1/9,1/9,1/9,1/36,1/36,1/36,1/36]; %Weighting factors.
ex = [0,1,-1,0,0,1,-1,-1,1]; %X-component of the particle velocity vector.
ey = [0,0,0,1,-1,1,1,-1,-1]; %Y-component of the particle velocity vector.
%-------INITIALIZATION OF THE VARIABLES-------
Ux = zeros(Ny,Nx); %X-component of the fluid velocity vector.
Uy = zeros(Ny,Nx); %Y-component of the fluid velocity vector.
rho = zeros(Ny,Nx); %Fluid density.
Uexact = zeros(Ny,Nx); %X-component of the exact fluid velocity.
Vexact = zeros(Ny,Nx); %Y-component of the exact fluid velocity.
feq = zeros(Ny,Nx,Nl); %Equilibrium distribution function.
f = zeros(Ny,Nx,Nl); %Particle distribution function.
orow = zeros(1,Nl); % offset for streaming, row (y)
ocol = zeros(1,Nl); % offset for streaming, col (x)
%-------INITIALIZATION OF THE SIMULATION (t=0)-------
for j = 1:Ny
    for i = 1:Nx
        Ux(j,i) = -Ulat*cos(Kx*(i-1))*sin(Kx*(j-1));
        Uy(j,i) = Ulat*sin(Kx*(i-1))*cos(Kx*(j-1));
        P = Ro*CsSquare-0.25*Ulat^2*(cos(2*Kx*(i-1))+cos(2*Kx*(j-1))); %Fluid pressure in lattice units.
        rho(j,i) = P/CsSquare;
    end
end
% sum(rho(:))
for k = 1:Nl
    term1 = ex(k)*Ux+ey(k)*Uy;
    term2 = Ux.^2+Uy.^2;
    feq(1:Ny,1:Nx,k) = weight(k)*rho(:,:).*(1+3*term1(:,:)+4.5*term1(:,:).^2-1.5*term2(:,:));
end
f(:,:,:) = feq(:,:,:);
%  system('rm /tmp/LBM3.txt');
%  for k=1:Nl
%    % dlmwrite("/tmp/LBM3.txt", f(:,:,:));
%    save_file = f(:,:,k);
%    save('-ascii', '-append', "/tmp/LBM3.txt", 'save_file');
%    system('echo -n >> /tmp/LBM3.txt');
%  end
% quit
%-------MAIN LOOP-------
for t = 1:Tsim
    %-------COLLISION-------
    for k = 1:Nl
        f(1:Ny,1:Nx,k) = f(1:Ny,1:Nx,k)*(1-omega)+omega*feq(1:Ny,1:Nx,k);
    end
%      % debug
%      if t == 2
%        display('blub');
%        system('rm /tmp/LBM3.txt');
%        system('touch /tmp/LBM3.txt');
%        for k=1:Nl
%          % dlmwrite("/tmp/LBM3.txt", f(:,:,:));
%          save_file = f(:,:,k);
%          save('-ascii', '-append', '/tmp/LBM3.txt', 'save_file');
%          system('echo >> /tmp/LBM3.txt');
%        end
%        quit
%      end
    %-------STREAMING (including the periodic boundary conditions)-------
    for k = 1:Nl
      orow(k) = mod(orow(k)-ey(k),Ny);
      ocol(k) = mod(ocol(k)-ex(k),Nx);
    end
    %-------CALCULATION OF THE MACROSCOPIC VARIABLES-------
    for j = 1:Ny
        for i = 1:Nx 
            rsum = 0; %Counter for the fluid density calculation.
            usum = 0; %Counter for the calculation of the x-component of the fluid velocity.
            vsum = 0; %Counter for the calculation of the y-component of the fluid velocity.
            for k = 1:Nl
                % compute index
                nrow = mod(j-1+orow(k)+Ny,Ny)+1;
                ncol = mod(i-1+ocol(k)+Nx,Nx)+1;
           
                rsum = rsum+f(nrow,ncol,k);
                usum = usum+f(nrow,ncol,k)*ex(k);
                vsum = vsum+f(nrow,ncol,k)*ey(k);
            end
            rho(j,i) = rsum;
            Ux(j,i) = usum/rho(j,i);
            Uy(j,i) = vsum/rho(j,i);
        end
    end
%      % debug
%      if t == 1
%        display('blub');
%        system('rm /tmp/LBM3.txt');
%        system('touch /tmp/LBM3.txt');
%        save_file = rho;
%        save('-ascii', '-append', '/tmp/LBM3.txt', 'save_file');
%        system('echo >> /tmp/LBM3.txt');
%        quit
%      end
    %-------VISUALIZATION OF THE VELOCITY FIELD-------
%     U = reshape(sqrt(Ux.^2+Uy.^2),Ny,Nx);
%     imagesc(U); caxis([0 Ulat]); colorbar; axis equal; drawnow
    %-------CALCULATION OF THE EQUILIBRIUM DISTRIBUTION FUNCTION-------
    for k = 1:Nl
        term1 = ex(k)*Ux+ey(k)*Uy;
        term2 = Ux.^2+Uy.^2;
        for j = 1:Ny
          for i = 1:Nx
            % compute index
            nrow = mod(j-1+orow(k)+Ny,Ny)+1;
            ncol = mod(i-1+ocol(k)+Nx,Nx)+1;
            
            feq(nrow,ncol,k) = weight(k)*rho(j,i).*(1+3*term1(j,i)+4.5*term1(j,i).^2-1.5*term2(j,i));
          end
        end
    end
end
%-------EXACT SOLUTION-------
for j = 1:Ny
    for i = 1:Nx
        Uexact(j,i) = -Ulat*cos(Kx*(i-1))*sin(Kx*(j-1))*exp(-2*Kx^2*vlat*Tsim);
        Vexact(j,i) = +Ulat*sin(Kx*(i-1))*cos(Kx*(j-1))*exp(-2*Kx^2*vlat*Tsim);
    end
end
%-------ABSOLUTE L2-ERROR-------
esum = 0; %Counter for the error calculation.
for j = 1:Ny
    for i = 1:Nx
        esum = esum+(Ux(j,i)-Uexact(j,i))^2+(Uy(j,i)-Vexact(j,i))^2;
    end
end
AbsL2Error = sqrt(esum/(Nx*Ny))