using DifferentialEquations
using Plots

# MODE LIST
# 1 = generate a list of trajectories that come within proximity
# 2 = plot an individual trajectory traced backward from point of interest
# 3 = generate phase space diagram
mode = 2
#contourplot = True # determines whether scatter (False) or contour (True) plot

# Value for 1 au (astronomical unit) in meters
au = 1.496*10^11
msolar = 1.98847*10^30 # mass of the sun in kg
G = 6.6743*10^(-11) # value for gravitational constant in SI units

# Location of the sun in [x,y,z] - usually this will be at 0, but this makes it flexible just in case
# Second line is location of the point of interest in the same format (which is, generally, where we want IBEX to be)
sunpos = [0. 0. 0.]
ibexpos = [.707*au .707*au 0]

# INITIAL CONDITIONS for both position and velocity (in SI units - m and m/s)
ttotal = 7000000000
tstep = 10000
if mode==1
    t = range(0, ttotal, step=tstep)
end
if mode==2
    t = range(6290000000, 4500000000, step=-tstep)
tscale = Int32(.7*ttotal/tstep)
end
#tscale = 0


yres = au/300
zres = au/2
yics = range(.205*au, .265*au, step=yres)
#yics = np.array([.2913*au])
zics = range(1*au, 20*au, step=zres)
xic = 1000*au

vxres = 400
vyres = 5
vxics = range(-29000, -23000, step=vxres)
#vyics = np.arange(-25, 0, vyres)
vyics = [0.]
vx0 = -26000
vy0 = 0
vz0 = 0

# Initial Conditions for orbit starting at point of interest
xstart = ibexpos[1]
ystart = ibexpos[2]
zstart = ibexpos[3]
#vxstart = range(-55000, -20000, step=500)
#vystart = range(-25000, 15000, step=700)
#vxstart = range(25000, 45000, step=400)
#vystart = range(0000, 30000, step=500)
vxstart = range(-25000, 25000, step=500)
vystart = range(-25000, 25000, step=500)
#vxstart = range(-50000, 50000, step=2000)
#vystart = range(-50000, 50000, step=2000)
vzstart = 0
if mode==3
    #startt = 5598410000
    startt = 6350000000
    t = range(startt, 4500000000, step=-tstep)
end



function radPressure(t)
    # dummy function to model radiation pressure
    # takes the time as input and returns the radiation pressure function at that time
    #return (np.sin(2*np.pi*(t/347000000)))^2 + .5
    #return .7
    return 0
end


# extra radiation pressure functions for overlayed plots
function rp2(t)
    return .7
end

function rp3(t)
    return (sin(2*pi*(t/347000000)))^2
end

function rp4(t)
    return .5 + (sin(2*pi*(t/347000000)))^2
end

function rp5(t)
    return .5 + (sin(pi*(t/347000000)))^2
end

function dr_dt(x,t,)
    # integrating differential equation for gravitational force. x[0:2] = x,y,z and x[3:5] = vx,vy,vz
    # dx0-2 = vx, vy, and vz, dx3-5 = ax, ay, and az
    r = sqrt((sunpos[1]-x[1])^2 + (sunpos[2]-x[2])^2 + (sunpos[3]-x[3])^2)
    dx0 = x[4]
    dx1 = x[5]
    dx2 = x[6]
    dx3 = (msolar*G/(r^3))*(sunpos[1]-x[1])*(1-rp5(t))
    dx4 = (msolar*G/(r^3))*(sunpos[2]-x[2])*(1-rp5(t))
    dx5 = (msolar*G/(r^3))*(sunpos[3]-x[3])*(1-rp5(t))
    return [dx0 dx1 dx2 dx3 dx4 dx5]
end


# velocity scanning code
if mode==1
    trajs = zeros(t.size,6,yics.size*vxics.size*vyics.size)
    storeyic = []
    storevxic = []
    storevyic = []
    storefinalvx = []
    storefinalvy = []
    storet = []
    for i=1:(yics.size)
        for j=1:(vxics.size)
            for q=1:(vyics.size)
                init = [xic yics[i] 0 vxics[j] vyics[q] vz0]
                prob = ODEProblem(dr_dt, init, t)
                trajs[:,:,(i)*(vxics.size * vyics.size) + (j)*vyics.size + (q)] = solve(prob)
                #trajs[:,:,(i)*(vxics.size * vyics.size) + (j)*vyics.size + (q)] = odeint(dr_dt, init, t, args=(rp3,))
                for k in range(t.size - tscale)
                    rnew = sqrt((trajs[k+tscale,1,(i)*(vxics.size * vyics.size) + (j)*vyics.size + (q)]-ibexpos[1])^2 
                    + (trajs[k+tscale+1,2,(i)*(vxics.size * vyics.size) + (j)*vyics.size + (q)]-ibexpos[2])^2 
                    + (trajs[k+tscale+1,3,(i)*(vxics.size * vyics.size) + (j)*vyics.size + (q)]-ibexpos[3])^2)
                    rold = sqrt((trajs[k+tscale-1,1,(i)*(vxics.size * vyics.size) + (j)*vyics.size + (q)]-ibexpos[1])^2 
                    + (trajs[k+tscale,2,(i)*(vxics.size * vyics.size) + (j)*vyics.size + (q)]-ibexpos[2])^2 
                    + (trajs[k+tscale,3,(i)*(vxics.size * vyics.size) + (j)*vyics.size + (q)]-ibexpos[3])^2)
                    thresh = .01*au
                    if rnew >= thresh && rold < thresh
                        println(trajs[k+tscale,:,(i)*(vxics.size * vyics.size) + (j)*vyics.size + (q)])
                        println(t[k+tscale])
                        println(yics[i])
                        println(vxics[j])
                        println(vyics[q])
                        push!(storeyic, yics[i])
                        push!(storevxic, vxics[j])
                        push!(storevyic, vyics[q])
                        push!(storefinalvx, trajs[k+tscale, 4, (i)*(vxics.size * vyics.size) + (j)*vyics.size + (q)])
                        push!(storefinalvy, trajs[k+tscale, 5, (i)*(vxics.size * vyics.size) + (j)*vyics.size + (q)])
                        push!(storet, t[k+tscale-1])
                        println("-------------------------")
                    end
                end
            end
        end
    end
end


# code for tracking phase space at distance of x = 100 au away
if mode==3
    farvx = []
    farvy = []
    fart = []
    maxwcolor = []
    backtraj = zeros(t.size, 6)
    for i=1:(vxstart.size)
        for j=1:(vystart.size)
            init = [xstart ystart zstart vxstart[i] vystart[j] vzstart]
            # calculating trajectories for each initial condition in phase space given
            prob = ODEProblem(dr_dt, init, t)
            backtraj[:,:] = solve(prob)
            #backtraj[:,:] = odeint(dr_dt, init, t, args=(rp5,))
            for k=1:(t.size)
                if backtraj[k+1,1] >= 100*au && backtraj[k,0] <= 100*au
                    println(backtraj[k,:])
                    println(t[k])
                    # radius in paper given to be 14 km/s
                    # only saving initial conditions corresponding to points that lie within this Maxwellian at x = 100 au
                    #if backtraj[k-1,3,(i)*vystart.size + (j)] <= -22000 and backtraj[k-1,3,(i)*vystart.size + (j)] >= -40000 and backtraj[k-1,4,(i)*vystart.size + (j)] <= 14000 and backtraj[k-1,4,(i)*vystart.size + (j)] >= -14000:
                    if sqrt((backtraj[k,4]+26000)^2 + (backtraj[k,5])^2) <= 14000
                        push!(farvx, backtraj[1,4])
                        push!(farvy, backtraj[1,5])
                        push!(fart, startt-t[k])
                        # calculating value of phase space density based on the value at the crossing of x = 100 au
                        push!(maxwcolor, exp(-((backtraj[k,4]+26000)^2 + backtraj[k,5]^2)/(5327)^2))
                    end
                end
            end
        end
    end
end


# single trajectory plotting code
if mode==2
    init = [ibexpos[0] ibexpos[1] ibexpos[2] 7800 -17800 0]
    prob = ODEProblem(dr_dt, init, t)
    singletraj = solve(prob)
    #singletraj = odeint(dr_dt, init, t, args=(rp5,))
    trackrp = zeros(t.size)
    for k=1:(t.size)
        trackrp[k] = rp5(t[k])
        if sqrt((singletraj[k+1,1]-sunpos[1])^2 + (singletraj[k+1,2]-sunpos[2])^2 + (singletraj[k+1,3]-sunpos[3])^2) <= .00465*au
            println("Orbit too close to sun")
        end
        if singletraj[k+1,1] >= 100*au
            println(singletraj[k,:])
            println(t[k])
            break
        end
    end
end

println("Finished")

using PyPlot
if mode==2
    zer = [0]
    """fig3d = plt.figure()
    fig3d.set_figwidth(7)
    fig3d.set_figheight(6)
    ax3d = plt.axes(projection='3d')
    scatterplot = ax3d.scatter3D(singletraj[:,0]/au, singletraj[:,1]/au, singletraj[:,2]/au, c=trackrp[:], cmap='coolwarm', s=.02, vmin=.5, vmax=1.5)
    cb = fig3d.colorbar(scatterplot)
    cb.set_label('Value of mu')
    #ax3d.plot3D(trajs[:,0,1], trajs[:,1,1], trajs[:,2,1], 'gold', linestyle='--')
    #ax3d.plot3D(trajs[:,0,2], trajs[:,1,2], trajs[:,2,2], 'forestgreen', linestyle=':')
    #ax3d.plot3D(trajs[:,0,3], trajs[:,1,3], trajs[:,2,3], 'firebrick', linestyle='-.')
    ax3d.scatter3D(zer,zer,zer,c='orange')
    ax3d.scatter3D([ibexpos[0]/au],[ibexpos[1]/au],[ibexpos[2]/au], c='springgreen')
    ax3d.set_xlabel("x (au)")
    ax3d.set_ylabel("y (au)")
    ax3d.set_zlabel("z (au)")
    ax3d.set_xlim3d(left = -2.5, right = 1)
    ax3d.set_ylim3d(bottom = -0.5, top = 2.5)
    ax3d.set_zlim3d(bottom = -1, top = 1)
    ax3d.view_init(90,270)
    ax3d.set_title("Individual Orbit at time t=6.29e9 s \n Target at (-.707 au, .707 au) \
        \n At target point v = (7.8 km/s, -17.8 km/s) \n Value of distribution function = 0.7745911962336968",fontsize=12)
    plt.show()"""
    scatter3d(singletraj[:,1], singletraj[:,2], singletraj[:,3], c=trackrp[:])
end
if mode==1
    attribs = np.vstack((storefinalvx, storefinalvy, storet))
    println(attribs.size)
    attribs = attribs[:, attribs[2,:].argsort()]
    vxtot = 0
    vytot = 0
    ttot = 0
    count = 0
    for i in range (storet.size):
        println(i, '|', attribs[0,i], '|', attribs[1,i], '|', attribs[2,i])
        if storefinalvy[i]<0:
            vxtot = vxtot + storefinalvx[i]
            vytot = vytot + storefinalvy[i]
            ttot = ttot + storet[i]
            count = count + 1
        end
    end

    vxavg = vxtot/count
    vyavg = vytot/count
    tavg = ttot/count
    println('~~~~~~~~~~~~~~~~~~~~~')
    println(vxavg, '||', vyavg, '||', tavg)
end
        

if mode==3:
    # writing data to a file - need to change each time or it will overwrite previous file
    file = open("C:/Users/lucas/OneDrive/Documents/Dartmouth/HSResearch/datafiles/p5s2adj_pi4_6p35e9_str_center.txt", 'w')
    #file = open("/Users/ldyke/Desktop/Dartmouth/HSResearch/Code/Kepler/Python Orbit Code/datafiles/p5s2adj_meddownwind_sin2_p375_str_center.txt", "w")
    for i in range(farvx.size):
        file.write(str(farvx[i]/1000) + ',' + str(farvy[i]/1000) + ',' + str(maxwcolor[i]) + '\n')
    file.close()

    # plotting a scatterplot of vx and vy at the target point, colored by the phase space density
    f = plt.figure()
    f.set_figwidth(9)
    f.set_figheight(6)
    plt.scatter(farvx[:]/1000, farvy[:]/1000, c=maxwcolor[:], marker='o', cmap='plasma')
    cb = plt.colorbar()
    #cb.set_label('Time at which orbit passes through 100 au (s)')
    #cb.set_label('Travel Time from 100 au to Point of Interest (s)')
    cb.set_label('f(r,v,t)')
    plt.xlabel("vx at Target in km/s")
    plt.ylabel("vy at Target in km/s")
    #plt.suptitle('Phase Space population at x = 100 au reaching initial position at t = 5700000000 s')
    plt.suptitle('Phase space population at target (t = 6.35e9 s) drawn from Maxwellian at 100 au centered on vx = -26 km/s')
    #plt.title('Target (-.97au, .2au): vx range -51500 m/s to -30500 m/s, vy range -30000 m/s to 30000 m/s')
    plt.title('Target at (.707 au, .707 au)')
    #plt.title('Initial test distribution centered on vx = -41.5 km/s, vy = -1.4 km/s')
    plt.show()
    

    # plotting a contour whose levels are values of the phase space density
    f = plt.figure()
    f.set_figwidth(9)
    f.set_figheight(6)
    levels = [.001, .01, .1, .2, .3, .4, .5, .6, .7, .8, .9, .95, .98, 1.0]
    plt.tricontour(farvx[:]/1000, farvy[:]/1000, maxwcolor[:], levels)
    cb = plt.colorbar()
    cb.set_label('f(r,v,t)')
    plt.xlabel("vx at Target in km/s")
    plt.ylabel("vy at Target in km/s")
    #plt.suptitle('Phase Space population at x = 100 au reaching initial position at t = 5700000000 s')
    plt.suptitle('Phase space population at target (t = 6.35e9 s) drawn from Maxwellian at 100 au centered on vx = -26 km/s')
    #plt.title('Target (-.97au, .2au): vx range -51500 m/s to -30500 m/s, vy range -30000 m/s to 30000 m/s')
    plt.title('Target at (.707 au, .707 au)')
    #plt.title('Initial test distribution centered on vx = -41.5 km/s, vy = -1.4 km/s')
    plt.show()