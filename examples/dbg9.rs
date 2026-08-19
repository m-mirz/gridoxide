fn main(){
    let n = gridoxide::ucte::read("tests/data/ucte/TestCase12Nodes.uct").unwrap();
    let c = n.tap_changers[0].as_ref().unwrap();
    let v: Vec<String> = [0,1,2,4,8,16].iter().map(|t| format!("{t}: {:.5}", -c.angle_deg(*t).unwrap())).collect();
    println!("gridoxide (negated): {{{}}}", v.join(", "));
}
